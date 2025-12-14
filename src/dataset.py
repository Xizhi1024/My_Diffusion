import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from PIL import Image
import glob
import os
import random


class MedicalImageDataset(Dataset):
    def __init__(self, pet_folder, ct_folder, image_size=128, augment=False, normalize=True):
        self.pet_paths = sorted(glob.glob(os.path.join(pet_folder, '*.*')))
        self.ct_paths = sorted(glob.glob(os.path.join(ct_folder, '*.*')))

        assert len(self.pet_paths) == len(self.ct_paths), "PET and CT folders must have same number of images"
        assert len(self.pet_paths) > 0, "Empty folders"

        self.image_size = image_size
        self.normalize = normalize

        transform_list = [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
        ]

        if augment:
            transform_list.extend([
                T.RandomHorizontalFlip(p=0.5),
            ])

        if normalize:
            transform_list.append(T.Normalize(mean=[0.5], std=[0.5]))

        self.transform = T.Compose(transform_list)

    def __len__(self):
        return len(self.pet_paths)

    def __getitem__(self, idx):
        pet_img = Image.open(self.pet_paths[idx]).convert('L')
        ct_img = Image.open(self.ct_paths[idx]).convert('L')

        pet_tensor = self.transform(pet_img)
        ct_tensor = self.transform(ct_img)

        return {
            'pet': pet_tensor,
            'ct': ct_tensor
        }


class PairedDataset(Dataset):
    def __init__(self, pet_folder, ct_folder, image_size=128, augment=False, normalize=True):
        self.dataset = MedicalImageDataset(pet_folder, ct_folder, image_size, augment, normalize)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        return torch.cat([data['pet'], data['ct']], dim=0)


def get_dataloaders(config):
    train_dataset = PairedDataset(
        pet_folder=config['train_pet_path'],
        ct_folder=config['train_ct_path'],
        image_size=config['image_size'],
        augment=config.get('augment', True),
        normalize=config.get('normalize', True)
    )

    val_dataset = PairedDataset(
        pet_folder=config['val_pet_path'],
        ct_folder=config['val_ct_path'],
        image_size=config['image_size'],
        augment=False,
        normalize=config.get('normalize', True)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config.get('num_workers', 4),
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get('val_batch_size', config['batch_size']),
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        pin_memory=True
    )

    return train_loader, val_loader