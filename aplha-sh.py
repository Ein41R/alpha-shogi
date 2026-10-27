import torch
from torch import nn

"""
    PIPELINE

    SFEN (1,32) -> EMB (1,128) -> ?some forward passes/mb attention -> expansion (1,88*81*2) -> masking -> softmax -> argmax -> (1,88*81*2) -> reverse mapping to piece->position 

    TODO: MCTS with upper confidence bound (UCB)
"""

class alphaShogiConfig:
    n_emb: int = 128#256
    dropout: float = 0.3
    n_src: int = 32
    n_tgt: int = 88*81*2 #softmax on this, deterministicly reverse to piece->position mapping


class alphaShogi():
    def __init__(self, config):
        super().__init__()
        self.config = config
        D = config.n_emb

        self.enc = nn.Sequential(
            nn.LayerNorm(D),
            nn.Linear(config.n_src, D)
        )

        self.fwd = nn.Sequential(
            nn.LayerNorm(D),
            nn.Linear(D, D),
            nn.ReLU(),
            nn.LayerNorm(D),
            nn.Linear(D, D),
            nn.ReLU()
        )

        self.dec = nn.Linear(D, config.n_tgt)

        self.Wq = nn.Linear(D, D)
        self.Wk = nn.Linear(D, D)
        self.Wv = nn.Linear(D, D)

    #We need attention to the previous moves if we do Tabula Rasa    
    def attn(self, x):
