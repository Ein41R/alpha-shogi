from config.trainConfig import (
    LOG_FREQ,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY
)

from config.torchConfig import (
    #ptdtype,
    dtype,
    device,
    ctx
)

import torch
import torch.nn
import torch.nn.functional as F

import cshogi

from util.sfenParser import Parser
import util.visualize_positions as viz

from model.alphaShogi import alphaShogi, alphaShogiConfig
import os


scaler = torch.amp.GradScaler(device=device.type, enabled=(dtype == "float16"))
torch.manual_seed(42)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
print(f"Using device: {device} with dtype {dtype}")


INPUT_FILEPATH = "data/"

def bitwise_loss(src_int, tgt_int) -> int: #which coincidentally is the hamming distance
    return (src_int ^ tgt_int).bit_count()

if __name__ == "__main__":
    model = alphaShogi(alphaShogiConfig()).to(device=device, dtype=dtype)
    model = torch.compile(model)

    optim = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.95), weight_decay=WEIGHT_DECAY)
    parser = Parser()
    board = cshogi.Board()

    loss_acc = 0

    for i in range(EPOCHS):
        for bin in os.listdir(INPUT_FILEPATH):
            if bin.endswith(".bin"):
                print(f"Loading {bin}...")
                parsefile = parser.parseFile(os.path.join(INPUT_FILEPATH, bin))
                sfens = []
                loss_acc = 0
                for entry in parsefile:
                    sfen = parser.decodeSfen(entry["sfen"])
                    sfens.append(sfen)
                for i in range(len(sfens)-1):
                    if i % LOG_FREQ == 0:
                        print(viz.render_hands(board))
                        print(viz.render_board(board))
                    board.set_psfen(sfen[i+1])
                    #sfen is array of 32 ints
                    sfen_tensor = torch.tensor(sfen, dtype=torch.int64, device=device).unsqueeze(0) # B,32
                    x = model.forward(sfen_tensor) # B,88*81*2
                    predicted_move = model.generate(sfen_tensor, board) # B,88*81*2 -> B, int32
                    target_move = board.peek()

                    #calc loss
                    loss = bitwise_loss(predicted_move[0,-1].item(), target_move)
                    scaler.scale(loss).backward()
                    scaler.step(optim)
                    scaler.update()

                    loss_acc += loss.item()
                print(f"Average loss for {bin}: {loss_acc/len(sfens)}")
