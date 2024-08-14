import os
import logging
from termcolor import colored

def setup_logger():
    working_dir = os.getcwd()
    vscode_path = os.path.join(working_dir, ".vscode")
    
    if os.path.exists(vscode_path):
        log_file_path = os.path.join(vscode_path, 'cli-agent.log')
    else:
        usr_dir = os.path.expanduser('~/.local/share')
        logs_path = os.path.join(usr_dir, 'cli-agent', 'logs')
        os.makedirs(logs_path, exist_ok=True)
        log_file_path = os.path.join(logs_path, 'cli-agent.log')

    # Create a logger
    logger = logging.getLogger('cli_agent')
    logger.setLevel(logging.DEBUG)  # Set to DEBUG to capture all levels

    # Create file handler which logs even debug messages
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.DEBUG)

    # Create console handler with a higher log level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.CRITICAL)

    # Create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    print(colored(f"Log is being written to {log_file_path}", 'yellow'))

    return logger

# Create and configure the logger
logger = setup_logger()