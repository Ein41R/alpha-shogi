import torch
from torch import nn
import numpy as np
import cshogi
from util.logger import logger

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

    # So this is ass, idk why i wrote this tbh, leaving it for now tho
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

    def cordsToInt32(self, from_sq, to_sq, is_promotion):
        return (from_sq << 7) | to_sq | (1 << 14 if is_promotion else 0)

    def esfToInt32(self, esf):
        # input is argmax output assuminb b = 1: 0,from,to,promotion
        from_sq = esf[1]
        to_sq = esf[2]
        is_promotion = True if esf[3] == 1 else False
        return (from_sq << 7) | to_sq | (1 << 14 if is_promotion else 0)


    def mask_tensor(self, tensor, board: cshogi.Board):
        # tensor: [B, 88*81*2]
        tensor = tensor.reshape(-1, 88, 81, 2)

        # cshogi stores captured-piece metadata above bit 14. The model action
        # space contains only source, destination, and promotion, so discard
        # that metadata when matching legal moves to action indices.
        legal_actions = set()
        for move in board.legal_moves:
            move = int(move) & 0x7FFF
            to_sq = move & 0x7F
            from_sq = (move >> 7) & 0x7F
            promote = (move >> 14) & 1
            if from_sq < 88 and to_sq < 81:
                legal_actions.add((from_sq, to_sq, promote))

        if not legal_actions:
            raise ValueError("Cannot mask move scores: board has no legal moves")

        legal_mask = torch.zeros(
            (88, 81, 2), dtype=torch.bool, device=tensor.device
        )
        for from_sq, to_sq, promote in legal_actions:
            legal_mask[from_sq, to_sq, promote] = True

        return tensor.masked_fill(~legal_mask.unsqueeze(0), -torch.inf)
    
