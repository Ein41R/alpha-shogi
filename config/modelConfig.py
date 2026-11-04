"""
configuring constant variables from project

"""

class alphaShogiConfig:
    n_emb: int = 128#256
    dropout: float = 0.3
    n_src: int = 32
    n_tgt: int = 88*81*2 #softmax on this, deterministicly reverse to piece->position mapping
    topk: int = 0 #topk moves to consider for MCTS, if 1 then argmax only
