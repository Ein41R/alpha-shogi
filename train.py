from config.trainConfig import (
    LOG_FREQ,
    EPOCHS
)

from config.torchConfig import (
    pdtype,
    dtype,
    device,
    ctx
)

import torch
import torch.nn
import torch.nn.functional as F

import cshogi

from util.sfenParser import Parser

from model.alphaShogi import alphaShogi, alphaShogiConfig
import os


scaler = torch.amp.GradScaler(device=device.type, enabled=(dtype == "float16"))
torch.manual_seed(42)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
print(f"Using device: {device} with dtype {dtype}")


INPUT_FILEPATH = "data/"


if __name__ == "__main__":
    model = alphaShogi(alphaShogiConfig()).to(device=device, dtype=dtype)
    model = torch.compile(model)

    optim = torch.optim.AdamW(model.parameters(), lr=alphaShogiConfig().lr, betas=(0.9, 0.95), weight_decay=alphaShogiConfig().weight_decay)
    parser = Parser()
    board = cshogi.Board()

    for i in range(EPOCHS):
        for bin in os.abortable(os.listdir(INPUT_FILEPATH)):
            if bin.endswith(".bin"):
                print(f"Loading {bin}...")
                parsefile = parser.parseFile(os.path.join(INPUT_FILEPATH, bin))
                sfens = []
                for entry in parsefile:
                    sfen = parser.decodeSfen(entry["sfen"])
                    sfens.append(sfen)
                for i,sfen in enumerate(sfens):
                    board.set_psfen(sfen)
                    #sfen is array of 32 ints
                    sfen_tensor = torch.tensor(sfen, dtype=torch.int64, device=device).unsqueeze(0) # B,32
                    



                data = torch.load(os.path.join(INPUT_FILEPATH, bin), map_location=device)
                x = data["x"].to(device=device, dtype=dtype)
                y = data["y"].to(device=device, dtype=dtype)

                optim.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=device.type, dtype=dtype):
                    logits = model(x)
                    loss = F.cross_entropy(logits, y)

                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()
