# ALWAYS IMPORT with "import paths" !
# and access with "paths.CACHE_DIR" etc
# This is to make sure that set_cache_dir() is effective

import os

# Default values
CACHE_DIR = "./cache"
MODELS_CACHE_DIR = os.path.join(CACHE_DIR, "models")
DATASETS_CACHE_DIR = os.path.join(CACHE_DIR, "datasets")
OUTPUT_DIR = "./output"

def set_cache_dir(base_dir):
    global CACHE_DIR, MODELS_CACHE_DIR, DATASETS_CACHE_DIR
    CACHE_DIR = base_dir
    MODELS_CACHE_DIR = os.path.join(CACHE_DIR, "models")
    DATASETS_CACHE_DIR = os.path.join(CACHE_DIR, "datasets")

def ensure_directories():
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(MODELS_CACHE_DIR, exist_ok=True)
    os.makedirs(DATASETS_CACHE_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
