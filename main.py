import os
import cshogi
import time

from util.envMng import envLoader
from util.logger import configure_logger
from util.logger import logger
from util.sfenParser import Parser
from util.visualize_positions import render_board, render_hands

from config.miscConfig import LOG_LEVEL

from pathlib import Path


def main():
    configure_logger(LOG_LEVEL  )

    board = cshogi.Board()
    parser = Parser()
    eMn = envLoader()

    eMn.parseFiles()
    
    for bin in Path(os.environ.get("DATA_PATH")).glob("*.bin*"):
        parsefile = parser.parseFile(bin)
        logger.INFO(f"Parsed {len(parsefile)} entries from {bin}")
        for entry in parsefile:
            sfen = parser.decodeSfen(entry["sfen"])
            move = parser.decodeMove(entry["sfen"], entry["move"])
            # score = entry["score"]
            # ply = entry["gamePly"]
            # result = entry["gameResult"]
            board.set_psfen(entry["sfen"])
            os.system('clear')
            print(render_hands(board))
            print(render_board(board))
            print()
            time.sleep(0.5)
            # return


if __name__ == "__main__":
    main()