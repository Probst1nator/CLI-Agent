import typing
from enum import Enum
from dataclasses import dataclass
from typing import Optional, List

class ConversationState(Enum):
    IDLE = "idle"              # Waiting for wake word
    ACTIVE = "active"          # In conversation
    TIMEOUT = "timeout"        # Conversation timed out

@dataclass
class Conversation:
    wake_word: str
    timeout_duration: int  # seconds
    last_interaction_time: float
    state: ConversationState
    conversation_history: List[str]

class WakeWordHandler:
    def __init__(self, wake_word: str, timeout_duration: int = 30) -> None:
        self.conversation = Conversation(
            wake_word=wake_word,
            timeout_duration=timeout_duration,
            last_interaction_time=0.0,
            state=ConversationState.IDLE,
            conversation_history=[]
        )

    def process_input(self, user_input: str, current_time: float) -> Optional[str]:
        # Check for timeout
        if (self.conversation.state == ConversationState.ACTIVE and 
            current_time - self.conversation.last_interaction_time > self.conversation.timeout_duration):
            self.conversation.state = ConversationState.TIMEOUT
            return "Conversation timed out. Please say the wake word to start a new conversation."

        # Update last interaction time
        self.conversation.last_interaction_time = current_time

        # Process input based on current state
        if self.conversation.state == ConversationState.IDLE:
            if self.conversation.wake_word.lower() in user_input.lower():
                self.conversation.state = ConversationState.ACTIVE
                return "Hello! How can I help you?"
            return None

        elif self.conversation.state == ConversationState.ACTIVE:
            self.conversation.conversation_history.append(user_input)
            return f"Processing: {user_input}"

        elif self.conversation.state == ConversationState.TIMEOUT:
            if self.conversation.wake_word.lower() in user_input.lower():
                self.conversation.state = ConversationState.ACTIVE
                self.conversation.conversation_history = []
                return "Hello again! How can I help you?"
            return None

    def end_conversation(self) -> None:
        self.conversation.state = ConversationState.IDLE
        self.conversation.conversation_history = []

# Example usage
if __name__ == "__main__":
    import time
    
    handler = WakeWordHandler(wake_word="hey assistant", timeout_duration=5)
    
    # Simulate conversation
    inputs = [
        "hello there",                 # Should not respond
        "hey assistant, how are you?", # Should activate
        "what's the weather?",         # Should respond
        "goodbye",                     # Should respond
        time.sleep(6),                 # Simulate timeout
        "what time is it?",            # Should not respond
        "hey assistant, back again"    # Should reactivate
    ]
    
    current_time = time.time()
    for user_input in inputs:
        if isinstance(user_input, float):
            continue
        
        response = handler.process_input(user_input, current_time)
        if response:
            print(f"User: {user_input}")
            print(f"Assistant: {response}\n")
        else:
            print(f"User: {user_input}")
            print("Assistant: *waiting for wake word*\n")
        
        current_time = time.time()
