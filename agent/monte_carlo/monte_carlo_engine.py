"""
Monte Carlo Action Engine for CLI-Agent.

This module contains the Monte Carlo decision-making engine extracted from main.py,
including branch generation, selection algorithms, and code execution confirmation.
"""

import asyncio
import argparse
import logging
import threading
from collections import Counter
from typing import Dict, Any, List, Optional, Tuple

from termcolor import colored

from core.globals import g
from core.chat import Chat
from core.llm_router import LlmRouter
from agent.text_painter import TextStreamPainter

# Import dependencies from main.py context
try:
    from main import (
        get_extract_blocks, get_user_input_with_bindings, enhanced_permission_prompt,
        FallbackModelIterator, get_provider_emoji_and_name
    )
except ImportError:
    # Fallback imports for standalone usage
    pass


async def confirm_code_execution(
    args: argparse.Namespace, 
    code_to_execute: str, 
    utils_manager: 'UtilsManager', 
    input_event: Optional[asyncio.Event] = None,
    input_lock: Optional[threading.Lock] = None,
    shared_input: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Handles code execution confirmation with a resilient, multi-stage Guard council.
    It attempts to gather a minimum of 3 votes. If initial models fail, it dynamically
    fetches diverse fallback models to ensure a decision can be made.
    """
    # Fast-path for simple, non-destructive shell commands
    if not get_extract_blocks()(code_to_execute, "python"):
        always_permitted_bash = ["ls ", "pwd ", "cd ", "echo ", "print ", "cat ", "head ", "tail ", "grep ", "sed ", "awk ", "sort "]
        bash_code = "\n".join(get_extract_blocks()(code_to_execute, ["bash", "shell"]))
        bash_code_lines = [line for line in bash_code.split("\n") if line.strip() and not line.strip().startswith("#")]
        if bash_code_lines and all(any(line.strip().startswith(cmd) for cmd in always_permitted_bash) for line in bash_code_lines):
            logging.info(colored("✅ Code execution permitted automatically (safe command list)", "green"))
            return True

    # --- Handle Default Mode (single LLM with manual confirmation) ---
    if not args.monte_carlo:
        # Determine the actual tool name from the code content
        tool_name = "bash"  # Default to bash
        action_description = "Execute bash command"
        
        if get_extract_blocks()(code_to_execute, "python"):
            tool_name = "python"
            action_description = "Execute Python code"
        elif get_extract_blocks()(code_to_execute, ["bash", "shell"]):
            tool_name = "bash"
            action_description = "Execute bash command"
        
        return await enhanced_permission_prompt(
            args, 
            tool_name=tool_name,
            action_description=action_description,
            xml_content=code_to_execute,
            input_event=input_event,
            input_lock=input_lock,
            shared_input=shared_input,
            get_user_input_func=get_user_input_with_bindings
        )

    # --- Guard-based Logic ---
    llm_config = g.get_llm_config()
    guard_council = [key for key, data in llm_config.items() if data.get('selected') and data.get('guard', 0) > 0 for _ in range(data['guard'])]

    if not guard_council:
        logging.warning(colored("⚠️  No Guard models found. Using manual confirmation.", "yellow"))
        user_input, _ = await get_user_input_with_bindings(
            args, None, 
            colored("Execute this code? (y=execute, n=abort): ", 'yellow'),
            input_event=input_event,
            input_lock=input_lock,
            shared_input=shared_input
        )
        return user_input.lower() in ['y', 'yes']

    # Prepare fallback iterator
    fallback_iterator = FallbackModelIterator()
    
    votes = []
    attempted_models = set()
    
    # --- Use Diverse Models from the Guard Council ---
    target_minimum = min(3, len(guard_council))
    
    while len(votes) < target_minimum and (guard_council or fallback_iterator.has_more()):
        # Select next model from guard council or fallback
        if guard_council:
            guard_model = guard_council.pop(0)
        else:
            guard_model = fallback_iterator.get_next()
            if not guard_model:
                break
        
        if guard_model in attempted_models:
            continue
        
        attempted_models.add(guard_model)
        
        # Prepare the Guard prompt
        guard_prompt = f"""Analyze the following code for safety and appropriateness. Consider potential risks including:

1. **Destructive Operations**: File deletion, system modifications, network interference
2. **Security Risks**: Credential exposure, privilege escalation, malicious downloads
3. **Resource Impact**: Excessive CPU/memory usage, infinite loops, large file operations
4. **System Stability**: Service disruption, configuration changes, dependency conflicts

Code to analyze:
```
{code_to_execute.strip()}
```

Respond with exactly one word: **ALLOW** or **BLOCK**

If the code appears safe and constructive, respond: **ALLOW**
If the code poses any significant risk, respond: **BLOCK**"""

        try:
            llm_router = LlmRouter()
            response = await llm_router.generate_response(
                model=guard_model,
                messages=[{"role": "user", "content": guard_prompt}],
                temperature=0.1,
                max_tokens=10
            )
            
            # Extract vote from response
            response_text = response.strip().upper()
            if "ALLOW" in response_text:
                vote = "ALLOW"
                votes.append(vote)
                emoji, name = get_provider_emoji_and_name(guard_model, llm_router)
                logging.info(colored(f"🛡️  {emoji} {name}: {vote}", "green"))
            elif "BLOCK" in response_text:
                vote = "BLOCK" 
                votes.append(vote)
                emoji, name = get_provider_emoji_and_name(guard_model, llm_router)
                logging.info(colored(f"🛡️  {emoji} {name}: {vote}", "red"))
            else:
                emoji, name = get_provider_emoji_and_name(guard_model, llm_router)
                logging.warning(colored(f"🛡️  {emoji} {name}: Invalid response '{response_text}'", "yellow"))
        
        except Exception as e:
            emoji, name = get_provider_emoji_and_name(guard_model)
            logging.warning(colored(f"🛡️  {emoji} {name}: Error - {str(e)}", "yellow"))
    
    # --- Decision Logic ---
    if not votes:
        logging.warning(colored("⚠️  No valid Guard votes received. Using manual confirmation.", "yellow"))
        user_input, _ = await get_user_input_with_bindings(
            args, None, 
            colored("Execute this code? (y=execute, n=abort): ", 'yellow'),
            input_event=input_event,
            input_lock=input_lock,
            shared_input=shared_input
        )
        return user_input.lower() in ['y', 'yes']
    
    vote_counts = Counter(votes)
    allow_count = vote_counts.get("ALLOW", 0)
    block_count = vote_counts.get("BLOCK", 0)
    
    logging.info(colored(f"🗳️  Guard Council Results: {allow_count} ALLOW, {block_count} BLOCK", "cyan"))
    
    # Majority rule
    if allow_count > block_count:
        logging.info(colored("✅ Code execution PERMITTED by Guard Council", "green"))
        return True
    elif block_count > allow_count:
        logging.info(colored("❌ Code execution BLOCKED by Guard Council", "red"))
        return False
    else:
        # Tie - manual confirmation
        logging.info(colored("⚖️  Guard Council tied. Manual confirmation required.", "yellow"))
        user_input, _ = await get_user_input_with_bindings(
            args, None, 
            colored(f"Guard Council tied ({allow_count}-{block_count}). Execute anyway? (y=execute, n=abort): ", 'yellow'),
            input_event=input_event,
            input_lock=input_lock,
            shared_input=shared_input
        )
        return user_input.lower() in ['y', 'yes']


async def select_best_branch(
    branches: List[str], 
    branch_model_map: Dict[int, str], 
    args: argparse.Namespace, 
    context_chat: Chat,
    current_round: int = 1
) -> Tuple[str, str]:
    """
    Advanced democratic elimination system for Monte Carlo branch selection.
    
    Uses multiple voting rounds with different LLMs to eliminate poor candidates
    until only the best branch remains. Each round, the worst-performing branch
    is eliminated based on LLM votes.
    
    Returns:
        Tuple of (selected_branch, selected_model_key)
    """
    
    if len(branches) == 1:
        selected_model = branch_model_map.get(0, 'unknown')
        return branches[0], selected_model
    
    llm_config = g.get_llm_config()
    voting_models = [key for key, data in llm_config.items() if data.get('selected') and data.get('vote', 0) > 0 for _ in range(data['vote'])]
    
    if not voting_models:
        logging.warning(colored("⚠️  No voting models configured. Using first branch.", "yellow"))
        selected_model = branch_model_map.get(0, 'unknown')
        return branches[0], selected_model
    
    active_branches = list(enumerate(branches))
    fallback_iterator = FallbackModelIterator()
    
    # --- Multi-Round Elimination Process ---
    while len(active_branches) > 1:
        logging.info(colored(f"🗳️  Round {current_round}: {len(active_branches)} branches remaining", "cyan"))
        
        # Collect votes for this round
        branch_scores = Counter()
        voting_round_models = voting_models.copy()
        
        # Vote collection with fallback support
        votes_collected = 0
        attempted_models = set()
        
        while votes_collected < len(active_branches) and (voting_round_models or fallback_iterator.has_more()):
            # Select voting model
            if voting_round_models:
                voting_model = voting_round_models.pop(0)
            else:
                voting_model = fallback_iterator.get_next()
                if not voting_model:
                    break
            
            if voting_model in attempted_models:
                continue
                
            attempted_models.add(voting_model)
            
            # Construct voting prompt
            branch_display = []
            for i, (original_idx, branch_content) in enumerate(active_branches):
                branch_num = i + 1
                model_name = branch_model_map.get(original_idx, 'unknown')
                branch_display.append(f"**Branch {branch_num}** (from {model_name}):\n{branch_content.strip()}\n")
            
            branches_text = "\n---\n".join(branch_display)
            
            voting_prompt = f"""You are evaluating AI-generated responses for quality and effectiveness. 

Context from conversation:
{context_chat.get_last_messages(3)}

Please evaluate these {len(active_branches)} response branches:

{branches_text}

Rank them from BEST to WORST based on:
- **Accuracy & Correctness**: Factual accuracy and logical reasoning
- **Helpfulness**: How well it addresses the user's needs  
- **Clarity**: Clear, well-structured communication
- **Completeness**: Thorough coverage of the topic
- **Practicality**: Actionable and realistic advice

Respond with ONLY the branch numbers in order from best to worst.
Example: "3 1 2" (meaning Branch 3 is best, Branch 1 is second, Branch 2 is worst)"""

            try:
                llm_router = LlmRouter()
                response = await llm_router.generate_response(
                    model=voting_model,
                    messages=[{"role": "user", "content": voting_prompt}],
                    temperature=0.2,
                    max_tokens=50
                )
                
                # Parse ranking response
                import re
                numbers = re.findall(r'\b(\d+)\b', response.strip())
                
                if len(numbers) == len(active_branches):
                    # Valid ranking - assign points (higher points = better)
                    max_points = len(active_branches)
                    for rank, branch_num_str in enumerate(numbers):
                        try:
                            branch_num = int(branch_num_str) - 1  # Convert to 0-indexed
                            if 0 <= branch_num < len(active_branches):
                                points = max_points - rank
                                original_idx = active_branches[branch_num][0]
                                branch_scores[original_idx] += points
                        except (ValueError, IndexError):
                            continue
                    
                    emoji, name = get_provider_emoji_and_name(voting_model, llm_router)
                    ranking_display = " > ".join(numbers)
                    logging.info(colored(f"🗳️  {emoji} {name}: {ranking_display}", "blue"))
                    votes_collected += 1
                else:
                    emoji, name = get_provider_emoji_and_name(voting_model, llm_router)
                    logging.warning(colored(f"🗳️  {emoji} {name}: Invalid ranking format", "yellow"))
                    
            except Exception as e:
                emoji, name = get_provider_emoji_and_name(voting_model)
                logging.warning(colored(f"🗳️  {emoji} {name}: Voting error - {str(e)}", "yellow"))
        
        # --- Elimination Logic ---
        if not branch_scores:
            logging.warning(colored("⚠️  No valid votes collected. Using first remaining branch.", "yellow"))
            selected_branch_idx, selected_branch = active_branches[0]
            selected_model = branch_model_map.get(selected_branch_idx, 'unknown')
            return selected_branch, selected_model
        
        # Find branch with minimum score for elimination
        min_score = min(branch_scores.values())
        candidates_for_elimination = [idx for idx, score in branch_scores.items() if score == min_score]
        
        if len(candidates_for_elimination) == 1:
            # Clear elimination
            eliminated_branch_index = candidates_for_elimination[0]
            active_branches = [(idx, branch) for idx, branch in active_branches if idx != eliminated_branch_index]
            
            eliminated_model_name = branch_model_map.get(eliminated_branch_index, 'unknown')
            logging.info(colored(f"⚖️  Knockout: Branch {eliminated_branch_index} (", "yellow") + eliminated_model_name + colored(f") eliminated ({min_score} votes).", "yellow"))
        else:
            logging.info(colored(f"⚖️  Deadlock for last place ({min_score} votes). No candidate will be eliminated this round.", "cyan"))

        # F. Prepare for the next round
        current_round += 1
    
    # Return the final remaining branch
    final_branch_idx, final_branch = active_branches[0]
    selected_model = branch_model_map.get(final_branch_idx, 'unknown')
    
    emoji, name = get_provider_emoji_and_name(selected_model)
    logging.info(colored(f"🏆 Final Selection: {emoji} {name}", "green"))
    return final_branch, selected_model


async def generate_branch(
    model_key: str, 
    branch_index: int, 
    lock: asyncio.Lock, 
    context_chat: Chat, 
    painter: TextStreamPainter, 
    temperature: float, 
    base64_images: list, 
    llm_config: dict, 
    branch_model_map: dict, 
    num_branches: int
):
    """
    Generates a single Monte Carlo branch using the specified model.
    
    This function runs in parallel with other branch generations and produces
    one possible response from the given model, with proper error handling
    and fallback mechanisms.
    """
    try:
        # Register this branch's model
        async with lock:
            branch_model_map[branch_index] = model_key
        
        llm_router = LlmRouter()
        emoji, name = get_provider_emoji_and_name(model_key, llm_router)
        
        # Clone chat context to avoid contamination between branches
        branch_chat = Chat()
        branch_chat.messages = context_chat.messages.copy()
        
        # Generate response for this branch
        stream_handler = painter.create_branch_handler(branch_index, emoji, name, num_branches)
        
        # Set model-specific parameters
        model_temperature = temperature
        if model_key in llm_config:
            model_temperature = llm_config[model_key].get('temperature', temperature)
        
        # Generate the response
        response = await llm_router.generate_response(
            model=model_key,
            messages=branch_chat.get_messages_for_provider(model_key),
            temperature=model_temperature,
            base64_images=base64_images,
            stream_handler=stream_handler
        )
        
        return response.strip()
        
    except Exception as e:
        async with lock:
            emoji, name = get_provider_emoji_and_name(model_key)
            logging.error(colored(f"Branch {branch_index} ({emoji} {name}) failed: {str(e)}", "red"))
        
        # Return a fallback response indicating failure
        return f"[Branch {branch_index} generation failed: {str(e)}]"