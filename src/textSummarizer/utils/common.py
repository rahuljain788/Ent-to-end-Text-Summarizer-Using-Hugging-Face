import os
from box.exceptions import BoxValueError
import yaml
from src.textSummarizer.logging import logger
from src.textSummarizer.utils.DatabaseUtility import DatabaseUtility
from src.textSummarizer.constants import *
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any


# Context manager for the database connection
@ensure_annotations
def get_mysql_db():
    """
    Get a database connection.

    :return: Database connection
    """
    db = DatabaseUtility(config_path=CONFIG_FILE_PATH)
    try:
        conn = db.connect()
        logger.info("connection " + str(conn))
        return db
    except:
        db.close_connection()

@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """reads yaml file and returns

    Args:
        path_to_yaml (str): path like input

    Raises:
        ValueError: if yaml file is empty
        e: empty file

    Returns:
        ConfigBox: ConfigBox type
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(path_to_yaml)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully !!!!")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e


@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """create list of directories

    Args:
        path_to_directories (list): list of path of directories
        ignore_log (bool, optional): ignore if multiple dirs is to be created. Defaults to False.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")


@ensure_annotations
def get_size(path: Path) -> str:
    """get size in KB

    Args:
        path (Path): path of the file

    Returns:
        str: size in KB
    """
    size_in_kb = round(os.path.getsize(path) / 1024)
    return f"~ {size_in_kb} KB"

