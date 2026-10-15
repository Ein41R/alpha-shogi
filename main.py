from pathlib import Path
import os
from util.envMng import envLoader
from util.logger import configure_logger
from util.logger import logger

def main():
    configure_logger(0)
    eMn = envLoader()
    eMn.parseFiles()
    for bin in Path(os.environ["DATA_PATH"]).glob("*.bin"):
        print(bin)


if __name__ == "__main__":
    main()