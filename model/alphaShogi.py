from shutil import move
import torch
from torch import inf, nn
from util.formatter import Formatter
import cshogi
from config.modelConfig import alphaShogiConfig
from util.logger import logger

"""
    PIPELINE

    SFEN (1,32) -> EMB (1,128) -> ?some forward passes/mb attention -> expansion (1,88*81*2) -> masking -> softmax -> argmax -> (1,88*81*2) -> reverse mapping to piece->position 

    TODO: MCTS with upper confidence bound (UCB)
"""

class alphaShogi(nn.Module):
    def __init__(self, config: alphaShogiConfig):
        super().__init__()
        self.config = config
        D = config.n_emb

        self.initial_norm = nn.LayerNorm(config.n_src, elementwise_affine=False, bias=False)

        # self.embed = nn.Embedding(config.n_src, D) # B,32 -> B,128
        self.embed = nn.Linear(config.n_src, D, bias=False) # B,32 -> B,128
        self.ponder = nn.Linear(D, D, bias=False) # B,128 -> B,128
        self.ln = nn.LayerNorm(D, elementwise_affine=False, bias=False)
        self.drop = nn.Dropout(config.dropout)
        self.expand = nn.Linear(D, config.n_tgt, bias=False) # B,128 -> B,88*81*2
        self.activate = nn.ReLU()
        self.dtype = torch.float16

        self.formatter = Formatter()

        # self.Wq = nn.Linear(D, D)
        # self.Wk = nn.Linear(D, D)
        # self.Wv = nn.Linear(D, D)

        self.apply(self._init_weights)

    # We need attention to the previous moves if we do Tabula Rasa 
    # same moves in a row makes draw, so attention must be paid to them()
    def attn(self, x):
        pass #skip attention for now, it just increases need for samples which we dont have

    def forward(self, x): #x is array in sfen format, convert to np array/tensor
        x = x.to(dtype=self.embed.weight.dtype)
        x = self.initial_norm(x)
        x = self.embed(x) # B,32 -> B,128

        x = self.ponder(x) # B,128 -> B,12
        x = self.drop(x)
        x = self.activate(x)
        x = self.ln(x)

        x = self.expand(x) # B,128 -> B,88*81*2
        x = self.activate(x)
        # then mask and softmax 
        return x

    #WARNING: TODO: THIS ASSUMES THAT B = 1 NEED TO EXTEND THIS LATER!!!!
    def generate(self,x,board: cshogi.Board):
        x = self.forward(x) # B,88*81*2
        x = self.formatter.mask_tensor(x, board) # B,88,81,2        
        logger.DEBUG(f"[\x1b[38;5;251malphaShogi.generate\x1b[0m]: shape {x.shape}")
        x = torch.softmax(x, dim=-1) # B,88,81,2
        logger.DEBUG(f"[\x1b[38;5;251malphaShogi.generate\x1b[0m]: shape {x.shape}")

        #JESUS Christ PLEASE WORK (removing B at this point since -> assumption B = 1 in place)
        shape = x.shape[1:]                      # (88, 81, 2) - grab BEFORE flattening
        x = x.flatten(start_dim=1)               # B, 88*81*2
        y = torch.argmax(x, dim=-1)              # B

        from_piece, to_square, promote = torch.unravel_index(y, shape)

        logger.DEBUG(f"[\x1b[38;5;251malphaShogi.generate\x1b[0m]: from_piece: {from_piece}, to_square: {to_square}, promote: {promote}")

        move = self.formatter.cordsToInt32(from_piece.item(), to_square.item(), promote.item())
        logger.INFO(f"[\x1b[38;5;251malphaShogi.generate\x1b[0m]: Success!")
        return move


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
