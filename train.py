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
import cshogi
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from util.sfenParser import Parser
from util.logger import configure_logger
from util.logger import logger

from model.alphaShogi import alphaShogi, alphaShogiConfig
import os


torch.manual_seed(42)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
logger.INFO(f"Using device: {device} with dtype {dtype}")


INPUT_FILEPATH = "data/"
PLOT_FILEPATH = "training_plots/training_loss.png"

def sampleByStd(probs: torch.Tensor) -> list[int]:
    """
    Sample from probabilities by standard deviation of the legal-action probabilities.
    returns list of indecies

    Args:
        probs: A 1D tensor of probabilities for legal actions.
    """
    max = probs.max(dim=-1, keepdim=True).values
    std = probs.std(dim=-1, unbiased=False, keepdim=True)
    threshold = max - std
    legal_mask = probs >= threshold
    legal_probs = probs * legal_mask
    legal_probs_sum = legal_probs.sum(dim=-1, keepdim=True)
    legal_probs_normalized = legal_probs / legal_probs_sum


def save_training_plot(steps, hamming_means, baselines, policy_loss_means):
    """Save window-averaged training metrics without opening a GUI window."""
    os.makedirs(os.path.dirname(PLOT_FILEPATH), exist_ok=True)

    figure, (loss_axis, policy_axis) = plt.subplots(
        2, 1, figsize=(11, 8), sharex=True, constrained_layout=True
    )
    loss_axis.plot(steps, hamming_means, label="Mean Hamming distance")
    loss_axis.plot(steps, baselines, label="EMA baseline", alpha=0.8)
    loss_axis.set_title("Move prediction loss during training")
    loss_axis.set_ylabel("Hamming distance (lower is better)")
    loss_axis.grid(True, alpha=0.25)
    loss_axis.legend()

    policy_axis.plot(steps, policy_loss_means, color="tab:orange", label="Mean policy loss")
    policy_axis.set_xlabel("Training positions processed")
    policy_axis.set_ylabel("Policy loss")
    policy_axis.grid(True, alpha=0.25)
    policy_axis.legend()

    figure.savefig(PLOT_FILEPATH, dpi=150)
    plt.close(figure)

def bitwise_loss(src_int, tgt_int) -> int:
    """Hamming distance over action bits, excluding capture metadata."""
    action_mask = 0x7FFF
    return ((int(src_int) & action_mask) ^ (int(tgt_int) & action_mask)).bit_count()

if __name__ == "__main__":
    configure_logger("INFO")

    # Keep master parameters in float32. AdamW updates to float16 parameters can
    # overflow/underflow and turn the policy logits into NaN or infinity.
    model = alphaShogi(alphaShogiConfig()).to(device=device, dtype=torch.float32)
    model = torch.compile(model)

    optim = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.95), weight_decay=WEIGHT_DECAY)
    parser = Parser()
    board = cshogi.Board()

    baseline = 0.0
    baseline_momentum = 0.95
    global_step = 0
    plot_steps = []
    plot_hamming_means = []
    plot_baselines = []
    plot_policy_loss_means = []

    for epoch in range(EPOCHS):
        for filename in os.listdir(INPUT_FILEPATH):
            if filename.endswith(".bin"):
                logger.INFO(
                    f"Epoch {epoch + 1}/{EPOCHS} | loading {filename}"
                )
                parsefile = parser.parseFile(os.path.join(INPUT_FILEPATH, filename))
                loss_acc = 0
                interval_distance = 0
                record_count = len(parsefile)
                # Cap progress output at about 20 updates per file, while still
                # respecting LOG_FREQ for smaller files.
                report_every = max(1, LOG_FREQ, (record_count + 19) // 20)
                interval_count = 0
                interval_policy_loss = 0.0
                exact_move_count = 0

                for index, entry in enumerate(parsefile):
                    packed_sfen = entry["sfen"]
                    board.set_psfen(packed_sfen)
                    target_move = int(board.move_from_psv(int(entry["move"])))

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
                    legal_action_count = int(torch.isfinite(flat_logits).sum().item())
                    distribution = torch.distributions.Categorical(logits=flat_logits)

                    #IDEA: dont evaluate every move, in order to save compute, we 
                    # sample all the moves above a certain threshold.
                    # Then we do the hamming distance and advantage/baseline 
                    # calculation for each of them
                    #And our total loss will be each probability from the distribution
                    # scaled by the corresponding advantage.
                    #Additionally increase advantage for dist 0 move,
                    # in case it isnt found and lower everything else

                    #sampled_index = distribution.sample() #sample picks one move by prob
                    sampled_index = distribution.sample() #sample picks one move by prob

                    log_probability = distribution.log_prob(sampled_index).squeeze(0)

                    promotion = int(sampled_index.item() % 2)
                    to_square = int((sampled_index.item() // 2) % 81)
                    from_square = int(sampled_index.item() // (2 * 81))
                    predicted_move = model.formatter.cordsToInt32(
                        from_square, to_square, promotion
                    )
                    hamming_distance = bitwise_loss(predicted_move, target_move)

                    baseline_before_update = baseline
                    advantage = hamming_distance - baseline_before_update
                    policy_loss = log_probability * advantage

                    optim.zero_grad(set_to_none=True)
                    policy_loss.backward()
                    optim.step()

                    baseline = ( #baseline EMA approaches a optimal baseline && baseline somewhat acts like a bias
                        baseline_momentum * baseline
                        + (1 - baseline_momentum) * hamming_distance
                    )

                    #Telemitry
                    loss_acc += hamming_distance
                    interval_distance += hamming_distance
                    interval_policy_loss += float(policy_loss.detach().item())
                    interval_count += 1
                    exact_move_count += int(hamming_distance == 0)
                    global_step += 1
                    is_report_step = interval_count >= report_every
                    is_last_record = index + 1 == record_count
                    if is_report_step or is_last_record:
                        window_average = interval_distance / interval_count
                        policy_loss_average = interval_policy_loss / interval_count
                        cumulative_average = loss_acc / (index + 1)
                        logger.INFO(
                            f"Epoch {epoch + 1}/{EPOCHS} | {filename} | "
                            f"{index + 1}/{record_count} positions | "
                            f"Hamming window avg={window_average:.3f}, "
                            f"file avg={cumulative_average:.3f}, "
                            f"exact moves={exact_move_count}/{index + 1}, "
                            f"policy loss avg={policy_loss_average:.4f}"
                        )
                        plot_steps.append(global_step)
                        plot_hamming_means.append(window_average)
                        plot_baselines.append(baseline)
                        plot_policy_loss_means.append(policy_loss_average)
                        save_training_plot(
                            plot_steps,
                            plot_hamming_means,
                            plot_baselines,
                            plot_policy_loss_means,
                        )
                        interval_distance = 0
                        interval_policy_loss = 0.0
                        interval_count = 0

                logger.INFO(
                    f"Epoch {epoch + 1}/{EPOCHS} | {filename} complete | "
                    f"records={record_count} "
                    f"average_hamming={loss_acc / record_count:.3f} "
                    f"plot={PLOT_FILEPATH}"
                    if record_count
                    else f"Epoch {epoch + 1}/{EPOCHS} | {filename} contains no records"
                )
