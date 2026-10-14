from pathlib import Path
import os
from util.envMng import envLoader

def main():
    eMn = envLoader.envLoader()
    eMn.parseFiles()
    for bin in Path(os.environ["DATA_PATH"]).glob("*.bin"):
        print(bin)


if __name__ == "__main__":
    main()