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
from util.logger import configure_logger
from util.logger import logger

from model.alphaShogi import alphaShogi, alphaShogiConfig
import os


torch.manual_seed(42)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
logger.INFO(f"Using device: {device} with dtype {dtype}")


INPUT_FILEPATH = "data/"

def bitwise_loss(src_int, tgt_int) -> int:
    """Hamming distance over action bits, excluding capture metadata."""
    action_mask = 0x7FFF
    return ((int(src_int) & action_mask) ^ (int(tgt_int) & action_mask)).bit_count()

if __name__ == "__main__":
    configure_logger(0)

    # Keep master parameters in float32. AdamW updates to float16 parameters can
    # overflow/underflow and turn the policy logits into NaN or infinity.
    model = alphaShogi(alphaShogiConfig()).to(device=device, dtype=torch.float32)
    model = torch.compile(model)

    optim = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.95), weight_decay=WEIGHT_DECAY)
    parser = Parser()
    board = cshogi.Board()

    baseline = 0.0
    baseline_momentum = 0.95

    for epoch in range(EPOCHS):
        for filename in os.listdir(INPUT_FILEPATH):
            if filename.endswith(".bin"):
                logger.INFO(f"Loading {filename}...")
                parsefile = parser.parseFile(os.path.join(INPUT_FILEPATH, filename))
                loss_acc = 0

                for index, entry in enumerate(parsefile):
                    packed_sfen = entry["sfen"]
                    board.set_psfen(packed_sfen)
                    target_move = int(board.move_from_psv(int(entry["move"])))

                    if index % LOG_FREQ == 0:
                        logger.INFO("Current position:")
                        logger.INFO(viz.render_hands(board))
                        logger.INFO(viz.render_board(board))

                    sfen_tensor = torch.as_tensor(
                        packed_sfen, dtype=torch.float32, device=device
                    ).unsqueeze(0)

                    # Sample a legal move from the model's action distribution.
                    # The sampled move's exact Hamming distance is the reward;
                    # REINFORCE supplies a gradient through its log probability.
                    logits = model(sfen_tensor).reshape(1, 88, 81, 2)
                    if not torch.isfinite(logits).all():
                        raise RuntimeError(
                            f"Model produced non-finite logits at {filename} record {index}."
                        )
                    # mask_tensor writes illegal-action sentinels in place; clone
                    # first so the model's ReLU output remains intact for autograd.
                    masked_logits = model.formatter.mask_tensor(logits.clone(), board)
                    flat_logits = masked_logits.flatten(start_dim=1).float()
                    if not torch.isfinite(flat_logits).any(dim=1).all():
                        raise RuntimeError(
                            f"No legal actions remained after masking at "
                            f"{filename} record {index}; check the move encoding."
                        )
                    distribution = torch.distributions.Categorical(logits=flat_logits)
                    sampled_index = distribution.sample() #sample picks one move by prob
                    log_probability = distribution.log_prob(sampled_index).squeeze(0)

                    promotion = int(sampled_index.item() % 2)
                    to_square = int((sampled_index.item() // 2) % 81)
                    from_square = int(sampled_index.item() // (2 * 81))
                    predicted_move = model.formatter.cordsToInt32(
                        from_square, to_square, promotion
                    )
                    hamming_distance = bitwise_loss(predicted_move, target_move)

                    advantage = hamming_distance - baseline
                    policy_loss = log_probability * advantage

                    optim.zero_grad(set_to_none=True)
                    policy_loss.backward()
                    optim.step()

                    baseline = ( #baseline EMA approaches a optimal baseline && baseline somewhat acts like a bias
                        baseline_momentum * baseline
                        + (1 - baseline_momentum) * hamming_distance
                    )
                    loss_acc += hamming_distance
                    if index % LOG_FREQ == 0:
                        logger.INFO(
                            f"Hamming distance for entry {index}: {hamming_distance}"
                        )
