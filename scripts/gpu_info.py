"""Print CUDA/GPU information."""
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device count:   {torch.cuda.device_count()}")
    print(f"Device name:    {torch.cuda.get_device_name(0)}")
    print(f"bf16 support:   {torch.cuda.is_bf16_supported()}")
    print(f"Memory (GB):    {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}")
else:
    print("Device: CPU")
