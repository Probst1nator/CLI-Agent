# File: globals.py
import os
import shutil
from typing import List, Optional
import argparse

class Globals:
    args: Optional[argparse.Namespace] = None
    
    PROJ_DIR_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    PROJ_PERSISTENT_STORAGE_PATH = os.path.join(PROJ_DIR_PATH, '.cliagent')
    PROJ_TEMP_STORAGE_PATH = os.path.join(PROJ_PERSISTENT_STORAGE_PATH, 'temporary')
    PROJ_ENV_FILE_PATH = os.path.join(PROJ_DIR_PATH, '.env')
    FORCE_LOCAL: bool = False
    
    DYNAMIC_MODEL_LIMITS_PATH = os.path.join(PROJ_PERSISTENT_STORAGE_PATH, 'dynamic_model_limits.json')
    UNCONFIRMED_FINETUNING_PATH = os.path.join(PROJ_TEMP_STORAGE_PATH, 'unconfirmed_finetuning_data')
    CONFIRMED_FINETUNING_PATH = os.path.join(PROJ_PERSISTENT_STORAGE_PATH, 'confirmed_finetuning_data')

    os.makedirs(PROJ_PERSISTENT_STORAGE_PATH, exist_ok=True)
    
    if os.path.exists(PROJ_TEMP_STORAGE_PATH):
        shutil.rmtree(PROJ_TEMP_STORAGE_PATH)
    os.makedirs(PROJ_TEMP_STORAGE_PATH, exist_ok=True)
    
    


g = Globals()