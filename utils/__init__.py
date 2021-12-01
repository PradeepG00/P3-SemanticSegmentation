# TODO: add imports HERE defining dependencies of the `util` package
import os.path
from pathlib import Path

PROJECT_ROOT = Path(os.path.abspath(__file__)).parent.parent
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
# toggle for less verbose output
DEBUG = False


def check_mkdir(dir_name: str) -> None:
    """Utility function that creates a directory if the path does not exist

    :param dir_name: str
    :return:
    """
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
