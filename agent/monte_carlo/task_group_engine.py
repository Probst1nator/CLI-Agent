"""
Monte Carlo Action Engine with asyncio.TaskGroup for robust concurrency.

This module replaces the asyncio.gather usage in main.py with the more robust
asyncio.TaskGroup pattern for better error handling and cancellation.
"""

from __future__ import annotations
import asyncio
import sys
from typing import List, Dict, Any, Optional, TYPE_CHECKING
import logging

if TYPE_CHECKING:
    from core.chat import Chat
    from agent.text_painter.stream_painter import TextStreamPainter

# Check Python version for TaskGroup availability
TASKGROUP_AVAILABLE = sys.version_info >= (3, 11)

if TASKGROUP_AVAILABLE:
    from asyncio import TaskGroup
else:
    # Fallback implementation for older Python versions
    class TaskGroup:
        """Fallback TaskGroup implementation for Python < 3.11."""
        
        def __init__(self):
            self._tasks: List[asyncio.Task] = []
            
        async def __aenter__(self):
            return self
            
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            if exc_type:
                # Cancel all tasks if there was an exception
                for task in self._tasks:
                    task.cancel()
                    
            # Wait for all tasks to complete or be cancelled
            if self._tasks:
                await asyncio.gather(*self._tasks, return_exceptions=True)
                
        def create_task(self, coro):
            task = asyncio.create_task(coro)
            self._tasks.append(task)
            return task

class MonteCarloEngine:
    """
    Enhanced Monte Carlo Action Engine using TaskGroup for robust concurrency.
    
    This class manages the concurrent execution of multiple AI model branches
    with proper error handling and cancellation support.
    """
    
    def __init__(self, context: 'AppContext') -> None:
        self.context = context
        self.logger = logging.getLogger("monte_carlo")
        
    async def generate_branches(
        self,
        models_for_tasks: List[str],
        context_chat: Chat,
        text_stream_painter: TextStreamPainter,
        temperature: float,
        base64_images: List[str],
        llm_config: Dict[str, Any]
    ) -> List[str]:
        """
        Generate multiple response branches using TaskGroup for robust concurrency.
        
        Args:
            models_for_tasks: List of model keys to use for branch generation
            context_chat: Current chat context
            text_stream_painter: Painter for streaming output
            temperature: Temperature setting for generation
            base64_images: List of base64-encoded images
            llm_config: LLM configuration dictionary
            
        Returns:
            List of generated response branches
        """
        num_branches = len(models_for_tasks)
        branch_model_map: Dict[int, str] = {}
        
        self.logger.info(f"Generating {num_branches} branches with TaskGroup")
        
        try:
            if TASKGROUP_AVAILABLE:
                async with TaskGroup() as tg:
                    print_lock = asyncio.Lock()
                    
                    # Create tasks for each branch
                    tasks = []
                    for i, model_key in enumerate(models_for_tasks):
                        task = tg.create_task(
                            self._generate_single_branch(
                                model_key=model_key,
                                branch_index=i,
                                lock=print_lock,
                                context_chat=context_chat,
                                painter=text_stream_painter,
                                temperature=temperature,
                                base64_images=base64_images,
                                llm_config=llm_config,
                                branch_model_map=branch_model_map,
                                num_branches=num_branches
                            )
                        )
                        tasks.append(task)
                
                # Extract results after TaskGroup completes successfully
                branch_results = [task.result() for task in tasks if not task.cancelled()]
                
            else:
                # Fallback to gather for Python < 3.11
                self.logger.warning("Using asyncio.gather fallback (Python < 3.11)")
                print_lock = asyncio.Lock()
                
                tasks = [
                    self._generate_single_branch(
                        model_key=model_key,
                        branch_index=i,
                        lock=print_lock,
                        context_chat=context_chat,
                        painter=text_stream_painter,
                        temperature=temperature,
                        base64_images=base64_images,
                        llm_config=llm_config,
                        branch_model_map=branch_model_map,
                        num_branches=num_branches
                    ) for i, model_key in enumerate(models_for_tasks)
                ]
                
                branch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Filter out exceptions
                branch_results = [
                    result for result in branch_results 
                    if not isinstance(result, Exception)
                ]
            
            # Filter and return valid responses
            response_branches = [res for res in branch_results if res and res.strip()]
            
            self.logger.info(f"Generated {len(response_branches)} valid branches")
            return response_branches
            
        except Exception as e:
            self.logger.error(f"Branch generation failed: {e}")
            return []
            
    async def _generate_single_branch(
        self,
        model_key: str,
        branch_index: int,
        lock: asyncio.Lock,
        context_chat: Chat,
        painter: TextStreamPainter,
        temperature: float,
        base64_images: List[str],
        llm_config: Dict[str, Any],
        branch_model_map: Dict[int, str],
        num_branches: int
    ) -> Optional[str]:
        """Generate a single response branch."""
        try:
            # Import here to avoid circular imports
            from main import generate_branch
            
            result = await generate_branch(
                model_key=model_key,
                branch_index=branch_index,
                lock=lock,
                context_chat=context_chat,
                painter=painter,
                temperature=temperature,
                base64_images=base64_images,
                llm_config=llm_config,
                branch_model_map=branch_model_map,
                num_branches=num_branches
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Branch {branch_index} ({model_key}) failed: {e}")
            return None
            
    async def evaluate_branches_with_taskgroup(
        self,
        evaluator_council: List[str],
        prompt: str,
        temperature: float
    ) -> List[tuple]:
        """
        Evaluate branches using TaskGroup for robust voting.
        
        This replaces the asyncio.gather usage in select_best_branch.
        """
        if TASKGROUP_AVAILABLE:
            try:
                async with TaskGroup() as tg:
                    tasks = []
                    index_map = {}
                    
                    for model in evaluator_council:
                        task = tg.create_task(
                            self._get_single_vote(model, prompt, temperature, index_map)
                        )
                        tasks.append(task)
                
                # Extract results after TaskGroup completes
                results = [task.result() for task in tasks if not task.cancelled()]
                valid_votes = [res for res in results if res[0] is not None]
                
                return valid_votes
                
            except Exception as e:
                self.logger.error(f"Branch evaluation failed: {e}")
                return []
        else:
            # Fallback for older Python versions
            tasks = [
                self._get_single_vote(model, prompt, temperature, {})
                for model in evaluator_council
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            valid_votes = [
                res for res in results 
                if not isinstance(res, Exception) and res[0] is not None
            ]
            return valid_votes
            
    async def _get_single_vote(
        self,
        model: str,
        prompt: str,
        temperature: float,
        index_map: Dict[str, int]
    ) -> tuple:
        """Get a single vote from an evaluator model."""
        try:
            # Import here to avoid circular imports and implement the vote logic
            # This would contain the logic from get_vote in main.py
            from main import get_vote
            return await get_vote(model, prompt, temperature, index_map)
        except Exception as e:
            self.logger.error(f"Vote from {model} failed: {e}")
            return (None, "")
            
    async def execute_guard_council_with_taskgroup(
        self,
        guard_council: List[str],
        code_to_check: str
    ) -> List[Any]:
        """
        Execute guard council using TaskGroup for robust safety checking.
        
        This replaces the asyncio.gather usage in confirm_code_execution.
        """
        if TASKGROUP_AVAILABLE:
            try:
                async with TaskGroup() as tg:
                    tasks = []
                    
                    for model in guard_council:
                        task = tg.create_task(
                            self._get_single_verdict(model, code_to_check)
                        )
                        tasks.append(task)
                
                # Extract results after TaskGroup completes
                results = [task.result() for task in tasks if not task.cancelled()]
                successful_results = [res for res in results if res is not None]
                
                return successful_results
                
            except Exception as e:
                self.logger.error(f"Guard council execution failed: {e}")
                return []
        else:
            # Fallback for older Python versions
            tasks = [
                self._get_single_verdict(model, code_to_check)
                for model in guard_council
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            successful_results = [
                res for res in results 
                if not isinstance(res, Exception) and res is not None
            ]
            return successful_results
            
    async def _get_single_verdict(self, model: str, code_to_check: str) -> Optional[Any]:
        """Get a single verdict from a guard model."""
        try:
            # Import here to avoid circular imports and implement the verdict logic
            # This would contain the logic from get_verdict in main.py
            from main import get_verdict
            return await get_verdict(model, code_to_check)
        except Exception as e:
            self.logger.error(f"Verdict from {model} failed: {e}")
            return None