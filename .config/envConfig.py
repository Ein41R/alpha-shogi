from Pathlib import Path
import os

class envLoader():
    def __init__(self):
        self.Path = Path(".")

    def parseFiles(self):
        for file in self.Path.glob("*.env"):
            with open(file, "r") as f:
                for line in f:
                    if line.strip() and not line.startswith("#"):
                        key, value = line.strip().split("=", 1)
                        os.environ[key] = value

    