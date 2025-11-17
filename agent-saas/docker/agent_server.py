"""
Generic Agent Server Wrapper

Converts CLI-based agents (minimal_agent, developer_agent) to WebSocket/HTTP server mode.
This wrapper is copied into each agent's Docker image and runs the agent as a web service.
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import traceback

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/workspace/logs/agent.log')
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="CLI-Agent SaaS - Agent Server",
    description="WebSocket/HTTP server for agent communication",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Container Manager will handle auth
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AgentWrapper:
    """
    Wrapper that loads and manages the agent instance.

    Supports both MinimalAgent and DeveloperAgent classes.
    """

    def __init__(self):
        self.agent = None
        self.agent_class_name = os.getenv("AGENT_CLASS_NAME", "MinimalAgent")
        self.agent_module_name = os.getenv("AGENT_MODULE_NAME", "main")
        self.chat_history_path = Path("/workspace/data/chat_history.json")
        self._load_agent()

    def _load_agent(self):
        """Dynamically load the agent class"""
        try:
            # Import the agent module (main.py in agent directory)
            import importlib
            module = importlib.import_module(self.agent_module_name)

            # Get the agent class (MinimalAgent or DeveloperAgent)
            if hasattr(module, self.agent_class_name):
                agent_class = getattr(module, self.agent_class_name)
                self.agent = agent_class()
                logger.info(f"Loaded agent: {self.agent_class_name}")
            else:
                raise ValueError(f"Agent class '{self.agent_class_name}' not found in module '{self.agent_module_name}'")

        except Exception as e:
            logger.error(f"Failed to load agent: {e}")
            logger.error(traceback.format_exc())
            raise

    def process_message(self, user_message: str) -> str:
        """
        Process a user message through the agent.

        Args:
            user_message: User's message

        Returns:
            Agent's complete response (after all tool calls)
        """
        try:
            # Agent's process_message handles the full loop
            response = self.agent.process_message(user_message)

            # Save chat history
            self._save_chat_history()

            return response

        except Exception as e:
            logger.error(f"Error processing message: {e}")
            logger.error(traceback.format_exc())
            return f"Error: {str(e)}"

    async def process_message_streaming(self, user_message: str):
        """
        Process message with streaming support.

        This is a wrapper around the agent's process_message that yields chunks.
        Since the current agents don't support true streaming at the character level,
        we'll stream the response after each tool execution.

        Yields:
            Dict with response chunks
        """
        try:
            # Add user message to chat
            self.agent.chat.add_user_message(user_message)

            # Agent loop - continue until no more tool calls
            max_iterations = 10
            iteration = 0

            while iteration < max_iterations:
                iteration += 1

                # Get LLM response with streaming
                messages = self.agent.chat.to_dict_list()

                try:
                    response_stream = self.agent.llm_router.generate_completion(messages, stream=True)

                    # Stream chunks from LLM
                    full_response = ""
                    for chunk in response_stream:
                        full_response += chunk
                        yield {
                            "type": "llm_chunk",
                            "content": chunk,
                            "is_complete": False
                        }

                    response = full_response

                except Exception as e:
                    error_msg = f"Error calling LLM: {str(e)}"
                    logger.error(error_msg)
                    yield {
                        "type": "error",
                        "content": error_msg,
                        "is_complete": True
                    }
                    return

                # Add assistant response to chat
                self.agent.chat.add_assistant_message(response)

                # Parse and execute tool calls
                tool_calls = self.agent._parse_tool_calls(response)

                if not tool_calls:
                    # No tool calls - done
                    yield {
                        "type": "final_response",
                        "content": response,
                        "is_complete": True
                    }

                    # Save chat history
                    self._save_chat_history()
                    return

                # Execute all tool calls
                tool_results = []

                for tool_name, content, full_match in tool_calls:
                    # Notify about tool execution
                    yield {
                        "type": "tool_execution",
                        "tool_name": tool_name,
                        "content": f"Executing {tool_name}...",
                        "is_complete": False
                    }

                    result = self.agent._execute_tool(tool_name, content)

                    if result is not None:
                        tool_results.append(f"Result of {tool_name}:\n{result}")

                        # Stream tool result
                        yield {
                            "type": "tool_result",
                            "tool_name": tool_name,
                            "content": result,
                            "is_complete": False
                        }

                # If there are no visible results, continue loop
                if not tool_results:
                    continue

                # Add tool results as user message
                results_message = "\n\n".join(tool_results)
                self.agent.chat.add_user_message(results_message)

            # Max iterations reached
            error_msg = "Error: Maximum iterations reached. Agent may be stuck in a loop."
            yield {
                "type": "error",
                "content": error_msg,
                "is_complete": True
            }

        except Exception as e:
            logger.error(f"Error in streaming: {e}")
            logger.error(traceback.format_exc())
            yield {
                "type": "error",
                "content": f"Error: {str(e)}",
                "is_complete": True
            }

    def get_chat_history(self) -> list[Dict[str, Any]]:
        """Get current chat history"""
        return self.agent.chat.to_dict_list()

    def clear_chat_history(self):
        """Clear chat history"""
        self.agent.chat = self.agent.chat.__class__()  # Create new Chat instance
        self.agent._initialize_system_prompt()
        self._save_chat_history()

    def _save_chat_history(self):
        """Save chat history to disk"""
        try:
            history = self.get_chat_history()
            self.chat_history_path.parent.mkdir(parents=True, exist_ok=True)

            with open(self.chat_history_path, 'w') as f:
                json.dump({
                    "saved_at": datetime.utcnow().isoformat(),
                    "messages": history
                }, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save chat history: {e}")

    def _load_chat_history(self):
        """Load chat history from disk"""
        try:
            if self.chat_history_path.exists():
                with open(self.chat_history_path, 'r') as f:
                    data = json.load(f)
                    return data.get("messages", [])
        except Exception as e:
            logger.error(f"Failed to load chat history: {e}")
        return []


# Global agent wrapper instance
agent_wrapper = AgentWrapper()


# === API Endpoints ===

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return JSONResponse({
        "status": "healthy",
        "agent_class": agent_wrapper.agent_class_name,
        "tools_loaded": len(agent_wrapper.agent.tools) if agent_wrapper.agent else 0
    })


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Main WebSocket endpoint for agent communication.

    Protocol:
    Client → Server:
        {
            "type": "chat_message",
            "content": "User message here"
        }

    Server → Client (streaming):
        {
            "type": "llm_chunk" | "tool_execution" | "tool_result" | "final_response" | "error",
            "content": "...",
            "is_complete": true/false
        }
    """
    await websocket.accept()
    logger.info("WebSocket connection established")

    try:
        while True:
            # Receive message
            data = await websocket.receive_text()

            try:
                message = json.loads(data)
                msg_type = message.get("type")
                content = message.get("content", "")

                logger.info(f"Received message type: {msg_type}")

                if msg_type == "chat_message":
                    # Process message with streaming
                    async for chunk in agent_wrapper.process_message_streaming(content):
                        await websocket.send_json(chunk)

                elif msg_type == "get_history":
                    # Send chat history
                    history = agent_wrapper.get_chat_history()
                    await websocket.send_json({
                        "type": "chat_history",
                        "content": history,
                        "is_complete": True
                    })

                elif msg_type == "clear_history":
                    # Clear chat history
                    agent_wrapper.clear_chat_history()
                    await websocket.send_json({
                        "type": "history_cleared",
                        "content": "Chat history cleared",
                        "is_complete": True
                    })

                else:
                    await websocket.send_json({
                        "type": "error",
                        "content": f"Unknown message type: {msg_type}",
                        "is_complete": True
                    })

            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "content": "Invalid JSON",
                    "is_complete": True
                })
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                logger.error(traceback.format_exc())
                await websocket.send_json({
                    "type": "error",
                    "content": str(e),
                    "is_complete": True
                })

    except WebSocketDisconnect:
        logger.info("WebSocket connection closed")


@app.get("/api/chat/history")
async def get_chat_history():
    """Get chat history (HTTP endpoint)"""
    try:
        history = agent_wrapper.get_chat_history()
        return JSONResponse({
            "status": "success",
            "messages": history
        })
    except Exception as e:
        logger.error(f"Error getting chat history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat/clear")
async def clear_chat_history():
    """Clear chat history (HTTP endpoint)"""
    try:
        agent_wrapper.clear_chat_history()
        return JSONResponse({
            "status": "success",
            "message": "Chat history cleared"
        })
    except Exception as e:
        logger.error(f"Error clearing chat history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat/send")
async def send_message(message: dict):
    """
    Send message to agent (HTTP endpoint - non-streaming).

    Request body:
    {
        "content": "User message here"
    }
    """
    try:
        content = message.get("content", "")
        if not content:
            raise HTTPException(status_code=400, detail="Missing 'content' field")

        response = agent_wrapper.process_message(content)

        return JSONResponse({
            "status": "success",
            "response": response
        })
    except Exception as e:
        logger.error(f"Error sending message: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/info")
async def get_agent_info():
    """Get information about the loaded agent"""
    try:
        return JSONResponse({
            "status": "success",
            "agent_class": agent_wrapper.agent_class_name,
            "tools": list(agent_wrapper.agent.tools.keys()) if agent_wrapper.agent else [],
            "tool_count": len(agent_wrapper.agent.tools) if agent_wrapper.agent else 0
        })
    except Exception as e:
        logger.error(f"Error getting agent info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # Get port from environment (default 8000)
    port = int(os.getenv("AGENT_PORT", "8000"))

    logger.info(f"Starting Agent Server on port {port}")
    logger.info(f"Agent class: {agent_wrapper.agent_class_name}")
    logger.info(f"Tools loaded: {len(agent_wrapper.agent.tools) if agent_wrapper.agent else 0}")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
