import torch
from torch import nn
import numpy as np

"""
Primary purpose is to convert
im calling my format the ESF the "exhausting shogi format""
88 x 81 x 2 into usable int32 format to pass onto cshogi engine

# cshogi int32 consists of 
0-6 to_sq 0-80
7-13 from_sq 0-80
bit14 1 << 14 / promotion flag

"""

class Formatter():
    def __init__(self):
        pass

    # return move as int32
    def t_esfToInt32(self, esf_t):
        # input esf tensor of size [B, 88*81*2]
        parse = esf_t.reshape(-1,88,81,2) #-> [B,88,81,2]

        move_vector = torch.zeros(esf_t.shape[0],88*81*2, dtype=torch.int32)

        for b in range(parse.shape[0]):
            for i in range(esf_t.shape[1]):
                for j in range(esf_t.shape[2]):
                    if esf_t[i,j,0] > 0:
                        from_sq = i
                        to_sq = j
                        is_promotion = True if esf_t[i,j,1] > 0 else False 
                        move_vector[b, i*81+j] = (from_sq << 7) | to_sq | (1 << 14 if is_promotion else 0)
        
        return move_vector