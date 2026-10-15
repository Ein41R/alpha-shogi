from pathlib import Path
import os
from util.envMng import envLoader
from util.logger import configure_logger
from util.logger import logger
from util.sfenParser import Parser

def main():
    configure_logger(0)

    parser = Parser()

    eMn = envLoader()
    eMn.parseFiles()
    for bin in Path(os.environ.get("DATA_PATH")).glob("*.bin*"):
        parsefile = parser.parseFile(bin)
        logger.INFO(f"Parsed {len(parsefile)} entries from {bin}")
        for entry in parsefile:
            sfen = parser.decodeSfen(entry["sfen"])
            move = parser.decodeMove(entry["sfen"], entry["move"])
            logger.DEBUG(f"SFEN: {sfen}, Move: {move}, Score: {entry['score']}, Ply: {entry['gamePly']}, Result: {entry['gameResult']}")
            return


if __name__ == "__main__":
    main()