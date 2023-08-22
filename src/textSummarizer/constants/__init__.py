import os
from pathlib import Path
current_path = os.getcwd()
CONFIG_FILE_PATH = Path(os.path.join(current_path, "config/config.yaml"))
PARAMS_FILE_PATH = Path(os.path.join(current_path, "params.yaml"))