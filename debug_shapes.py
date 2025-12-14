import torch
from src.model.latent_diffusion import LatentDiffusionModel
from src.dataset import get_dataloaders
import yaml

# Load config
with open('configs/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Create model
model = LatentDiffusionModel(
    image_size=config['image_size'],
    latent_channels=config['latent_channels'],
    unet_config=config.get('unet_config', {})
)

# Create dataloader
train_loader, _ = get_dataloaders(config)

# Get a batch
batch = next(iter(train_loader))

# Extract PET and CT
if isinstance(batch, dict):
    pet = batch['pet']
    ct = batch['ct']
else:
    pet, ct = batch.chunk(2, dim=1)

print(f"Input shapes:")
print(f"  PET (condition): {pet.shape}")
print(f"  CT (target): {ct.shape}")

# Test encoding
pet_latent = model.encode_to_latent(pet)
ct_latent = model.encode_to_latent(ct)

print(f"\nLatent shapes:")
print(f"  PET latent: {pet_latent.shape}")
print(f"  CT latent: {ct_latent.shape}")

# Test concatenation
concat = torch.cat([ct_latent, pet_latent], dim=1)
print(f"\nConcatenated shape: {concat.shape}")

# Test UNet config
print(f"\nUNet config:")
print(f"  in_channels: {model.unet.config.in_channels}")
print(f"  out_channels: {model.unet.config.out_channels}")
print(f"  sample_size: {model.unet.config.sample_size}")

# Test forward pass with noise
noise = torch.randn_like(ct_latent)
timesteps = torch.randint(0, model.noise_scheduler.config.num_train_timesteps, (ct.shape[0],))

noisy_latents = model.add_noise(ct_latent, noise, timesteps)
print(f"\nNoisy latents shape: {noisy_latents.shape}")

# Test conditioning
condition_latent = model.encode_to_latent(pet)
if noisy_latents.shape[2:] != condition_latent.shape[2:]:
    print(f"Shape mismatch detected!")
    print(f"  Noisy latent spatial dims: {noisy_latents.shape[2:]}")
    print(f"  Condition latent spatial dims: {condition_latent.shape[2:]}")
else:
    print("Spatial dimensions match!")

# Test the actual problematic call
try:
    noise_pred = model.predict_noise(noisy_latents, timesteps, pet)
    print(f"\nSuccess! Noise prediction shape: {noise_pred.shape}")
except Exception as e:
    print(f"\nError during predict_noise: {e}")
    import traceback
    traceback.print_exc()