import torch
from contextlib import nullcontext

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

#device type
dtype = (
    "bfloat16"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else "float16"
)

#parameter type
ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[dtype]

#context manager for automatic mixed precision
ctx = (
    torch.amp.autocast(device_type=device.type, dtype=ptdtype)
    if "cuda" in device.type
    else nullcontext()
) 