"""
10/13/2026 by Ein41r
Python script for parsing data in YaneuraOu PackedSfenValue format.
"""

import numpy as np

PackedSfenValue_dtype = np.dtype(
    [
        ("sfen", np.uint8, 32),  # 32-byte Huffman/packed SFEN
        ("score", np.int16),  # 2-byte evaluation score
        ("move", np.uint16),  # 2-byte move encoding
        ("gamePly", np.uint16),  # 2-byte ply number
        ("gameResult", np.int8),  # 1-byte game result (1, -1, 0)
        ("padding", np.uint8),  # 1-byte struct alignment padding
    ]
)

class Parser():
    def __init__(self):
        self.dtype = PackedSfenValue_dtype

    def parseFile(self, file_path):
        self.file_path = file_path
        data = np.fromfile(self.file_path, dtype=PackedSfenValue_dtype)
        return data

    def parseDir(self, dir_path):
        self.dir_path = dir_path
        data_list = []
        for file in self.dir_path.glob("*.bin"):
            data = np.fromfile(file, dtype=PackedSfenValue_dtype)
            data_list.append(data)
        return np.concatenate(data_list)

    