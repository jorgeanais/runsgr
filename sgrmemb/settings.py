from configparser import ConfigParser, ExtendedInterpolation
from dataclasses import dataclass
import logging
from pathlib import Path

"""
This module contains the configuration of the project. It uses the configparser module
"""

CONFIG_FILE = "params.ini"
configparser = ConfigParser(interpolation=ExtendedInterpolation())
configparser.read(CONFIG_FILE)


files_config = configparser["Files"]


@dataclass
class FileConfig:
    """Stores the file configuration"""

    DATA_PATH: Path = Path(files_config["data_path"])
    OUTPUT_PATH: Path = Path(files_config["output_path"])

    TRAIN_DATA: Path = DATA_PATH / files_config["train_data"]
    TEST_DATA: Path = DATA_PATH / files_config["test_data"]
    VASILIEV_DATA: Path = DATA_PATH / files_config["vasiliev_data"]


params_config = configparser["Params"]


@dataclass
class ParamsConfig:
    """Stores the parameters configuration"""

    RECIPE_MODULE: str = ".".join(["sgrmemb", "recipes", params_config["script_name"]])
    MAX_ITERATIONS: int = int(params_config["max_iterations"])
    GMM_N_COMPONENTS: int = int(params_config["gmm_n_components"])
    TOLERANCE: float = float(params_config["tolerance"])


def set_logger():
    """Sets the logger configuration"""
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filename="logfile.txt",
        level=logging.INFO,
    )

    logging.info("Started")

    for f in configparser.items("Files"):
        logging.info(f)

    for p in configparser.items("Params"):
        logging.info(p)
