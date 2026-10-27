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


class alphaShogi(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        D = config.n_emb

        self.embed = nn.Embedding(config.n_src, D) # B,32 -> B,128
        self.ponder = nn.Linear(D, D, bias=False) # B,128 -> B,128
        self.ln = nn.LayerNorm(D, elementwise_affine=False, bias=False)
        self.drop = nn.Dropout(config.dropout)
        self.expand = nn.Linear(D, config.n_tgt, bias=False) # B,128 -> B,88*81*2

        # self.Wq = nn.Linear(D, D)
        # self.Wk = nn.Linear(D, D)
        # self.Wv = nn.Linear(D, D)

        self.apply(self._init_weights)

    # We need attention to the previous moves if we do Tabula Rasa 
    # same moves in a row makes draw, so attention must be paid to them()
    def attn(self, x):
        pass #skip attention for now, it just increases need for samples which we dont have

    def forward(self, x): #x is array in sfen format, convert to np array/tensor
        x = self.embed(x) # B,32 -> B,128
        x = self.ln(x)
        x = self.ponder(x) # B,128 -> B,128
        x = self.drop(x)
        x = self.ln(x)
        x = self.expand.T @ x.T # B,128 -> B,88*81*2
        return x

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

