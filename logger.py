import os
import logging
from termcolor import colored
from globals import g

def setup_logger():
    log_file_path = os.path.join(g.PROJ_VSCODE_DIR_PATH, 'cli-agent.log')
    
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