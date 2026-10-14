from setupPATH import setupPATH
setupPATH()

from sfenParser import Parser
from pathlib import Path
from envMng import envLoader
import os

def main():
    eMn = envLoader()
    eMn.parseFiles()
    for bin in Path(os.environ["DATA_PATH"]).glob("*.bin"):
        print(bin)


if __name__ == "__main__":
    main()