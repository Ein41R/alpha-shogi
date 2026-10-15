from pathlib import Path
from util.logger import logger
import os

class envLoader():
    def __init__(self):
        self.Path = Path(".")

    def parseFiles(self):

        for file in self.Path.glob("*.env*"):
            logger.DEBUG(f"Parsing env file: {file}")
            with open(file, "r") as f:
                for line in f:
                    if line.strip() and not line.startswith("#"):
                        key, value = line.strip().split("=", 1)
                        value = value.strip('"').strip("'")  # Remove quotes if present
                        os.environ[key] = value
                        logger.DEBUG(f"Set environment variable: {key}={value}")
