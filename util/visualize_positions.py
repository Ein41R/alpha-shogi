import argparse
import os
import sys
from pathlib import Path

import cshogi

from util.sfenParser import Parser


RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
GOLD = "\033[38;5;220m"
BOARD_LINE = "\033[38;5;94m"
SENTE = "\033[38;5;203m"
GOTE = "\033[38;5;117m"


def paint(text, style, enabled):
    if not enabled:
        return text
    return f"{style}{text}{RESET}"


def render_board(board, color=None):
    color = sys.stdout.isatty() if color is None else color
    symbols = cshogi.PIECE_JAPANESE_SYMBOLS
    files = "  " + "┬".join(f"{file_number:^4}" for file_number in range(9, 0, -1))
    rows = [paint(files, BOLD + GOLD, color)]
    rows.append(paint("  ┌" + "────┬" * 8 + "────┐", BOARD_LINE, color))

    for rank in range(9):
        cells = []
        for file_index in range(8, -1, -1):
            piece = board.pieces[rank * 9 + file_index]
            if piece == cshogi.NONE:
                cells.append("    ")
                continue
            owner = "▲" if piece < cshogi.WPAWN else "△"
            style = SENTE if piece < cshogi.WPAWN else GOTE
            cells.append(paint(f" {owner}{symbols[piece % 16]}", style, color))
        rows.append(
            paint(f"{rank + 1} │", BOLD + GOLD, color)
            + paint("│".join(cells), "", color)
            + paint("│", BOARD_LINE, color)
        )
        if rank < 8:
            rows.append(paint("  ├" + "────┼" * 8 + "────┤", BOARD_LINE, color))

    rows.append(paint("  └" + "────┴" * 8 + "────┘", BOARD_LINE, color))
    return "\n".join(rows)


def render_hands(board, color=None):
    enabled = sys.stdout.isatty() if color is None else color
    hands = []
    for hand_color, label in ((cshogi.WHITE, "後手持駒"), (cshogi.BLACK, "先手持駒")):
        pieces = []
        for symbol, count in zip(
            cshogi.HAND_PIECE_JAPANESE_SYMBOLS,
            board.pieces_in_hand[hand_color],
        ):
            if count:
                pieces.append(f"{symbol}{count}")
        hands.append(
            paint(
                f"{label}: " + (" ".join(pieces) if pieces else "なし"),
                GOTE if hand_color == cshogi.WHITE else SENTE,
                enabled,
            )
        )
    return "\n".join(hands)


def load_records(path):
    parser = Parser()
    if path.is_file():
        return parser.parseFile(path), parser

    files = sorted(path.glob("*.bin*"))
    if not files:
        raise FileNotFoundError(f"No .bin files found in {path}")
    return parser.parseFile(files[0]), parser


def main():
    argument_parser = argparse.ArgumentParser(
        description="Decode and visualize one YaneuraOu PackedSfenValue record."
    )
    argument_parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=os.environ.get("DATA_PATH"),
        help="PackedSfen file or directory; defaults to DATA_PATH.",
    )
    argument_parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Zero-based record index to display.",
    )
    argument_parser.add_argument(
        "--svg",
        type=Path,
        default=Path("visualizations/position.svg"),
        help="Output path for the SVG board.",
    )
    argument_parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI colors in terminal output.",
    )
    args = argument_parser.parse_args()

    if args.path is None:
        argument_parser.error("provide a PackedSfen file/directory or set DATA_PATH")

    records, parser = load_records(args.path)
    if not 0 <= args.index < len(records):
        argument_parser.error(f"index must be between 0 and {len(records) - 1}")

    entry = records[args.index]
    board = cshogi.Board()
    board.set_psfen(entry["sfen"])
    move = parser.decodeMove(entry["sfen"], entry["move"])

    args.svg.parent.mkdir(parents=True, exist_ok=True)
    args.svg.write_text(board.to_svg(), encoding="utf-8")

    use_color = not args.no_color
    print(render_hands(board, color=use_color))
    print(render_board(board, color=use_color))
    print()
    print(f"SFEN:   {board.sfen()}")
    print(f"Move:   {move}")
    print(f"Score:  {int(entry['score'])}")
    print(f"Ply:    {int(entry['gamePly'])}")
    print(f"Result: {int(entry['gameResult'])}")
    print(f"SVG:    {args.svg.resolve()}")


if __name__ == "__main__":
    main()
