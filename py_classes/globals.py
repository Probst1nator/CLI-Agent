# File: globals.py
import os
import shutil
from typing import List, Optional
import argparse
import logging

class Globals:
    args: Optional[argparse.Namespace] = None
    
    PROJ_DIR_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    PROJ_PERSISTENT_STORAGE_PATH = os.path.join(PROJ_DIR_PATH, '.cliagent')
    PROJ_TEMP_STORAGE_PATH = os.path.join(PROJ_PERSISTENT_STORAGE_PATH, 'temporary')
    PROJ_ENV_FILE_PATH = os.path.join(PROJ_DIR_PATH, '.env')
    FORCE_LOCAL: bool = False
    
    DYNAMIC_MODEL_LIMITS_PATH = os.path.join(PROJ_PERSISTENT_STORAGE_PATH, 'dynamic_model_limits.json')
    MODEL_RATE_LIMITS_PATH = os.path.join(PROJ_PERSISTENT_STORAGE_PATH, 'model_rate_limits.json')
    UNCONFIRMED_FINETUNING_PATH = os.path.join(PROJ_TEMP_STORAGE_PATH, 'unconfirmed_finetuning_data')
    CONFIRMED_FINETUNING_PATH = os.path.join(PROJ_PERSISTENT_STORAGE_PATH, 'confirmed_finetuning_data')

    os.makedirs(PROJ_PERSISTENT_STORAGE_PATH, exist_ok=True)
    
    if os.path.exists(PROJ_TEMP_STORAGE_PATH):
        shutil.rmtree(PROJ_TEMP_STORAGE_PATH)
    os.makedirs(PROJ_TEMP_STORAGE_PATH, exist_ok=True)
    
g = Globals()

def configure_logging():
    # Configure root logger 
    root_logger = logging.getLogger()
    
    # Clear any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.ERROR)  # Only show ERROR and above in console
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Create file handler for all logs
    os.makedirs(g.PROJ_PERSISTENT_STORAGE_PATH, exist_ok=True)
    log_file_path = os.path.join(g.PROJ_PERSISTENT_STORAGE_PATH, 'app.log') 
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)  # Still log INFO and above to file
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # Set overall logger level - this needs to be the lowest level you want to capture
    root_logger.setLevel(logging.INFO)
    
    # Silence specific loggers that are too verbose
    logging.getLogger('py_classes.cls_llm_router').setLevel(logging.ERROR)
    
    logging.info("Logging configured")

# Configure logging after Globals is instantiated
configure_logging()