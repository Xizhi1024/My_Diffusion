import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from PIL import Image
import glob
import os
import random


class MedicalImageDataset(Dataset):
    def __init__(self, pet_folder, ct_folder, image_size=128, augment=False, normalize=True,
                 use_medical_preprocessing=False, ct_window_width=400, ct_window_center=50,
                 pet_max_suv=None, pet_target_range="[-1,1]"):
        if use_medical_preprocessing:
            raise ValueError(
                "use_medical_preprocessing is not supported in this code path. "
                "Please set use_medical_preprocessing: false."
            )

        self.pet_paths = sorted(glob.glob(os.path.join(pet_folder, '*.*')))
        self.ct_paths = sorted(glob.glob(os.path.join(ct_folder, '*.*')))

        assert len(self.pet_paths) == len(self.ct_paths), "PET and CT folders must have same number of images"
        assert len(self.pet_paths) > 0, "Empty folders"

        self.image_size = image_size
        self.augment = augment
        # 注意：我们不再依赖外部的 normalize 参数或 medical preprocessor
        # 我们将在 __getitem__ 中手动执行鲁棒的 Min-Max 归一化

    def __len__(self):
        return len(self.pet_paths)

    def __getitem__(self, idx):
        # 1. 读取图像
        pet_img = Image.open(self.pet_paths[idx]).convert('L')
        ct_img = Image.open(self.ct_paths[idx]).convert('L')

        # 2. 转为 Tensor (0.0 - 1.0)
        # 【修复】与 diffusion 项目保持一致的处理流程
        # T.ToTensor() 会自动将 0-255 映射到 0.0-1.0
        transform_to_tensor = T.Compose([
            T.Resize((self.image_size, self.image_size)),
            T.CenterCrop(self.image_size),  # 【新增】添加 CenterCrop 确保尺寸精确
            T.ToTensor(),
        ])

        pet_tensor = transform_to_tensor(pet_img) # shape [1, 128, 128]
        ct_tensor = transform_to_tensor(ct_img)

        # 3. 数据增强 - 【修复】只保留水平翻转，移除复杂的 affine 变换
        # 参考 diffusion 项目：只使用简单的 RandomHorizontalFlip
        # 过度的 affine 变换（旋转、平移、缩放）可能引入插值伪影和不一致性
        if self.augment:
            if random.random() > 0.5:
                pet_tensor = T.functional.hflip(pet_tensor)
                ct_tensor = T.functional.hflip(ct_tensor)
            # 【移除】affine 变换（旋转、平移、缩放）
            # - 旋转和平移会产生插值伪影
            # - 医学图像对几何变形敏感
            # - 与 diffusion 项目保持一致

        # 4. 【修复】使用简单的线性归一化到 [-1, 1]
        # 与 diffusion 项目保持一致，保持原始像素分布
        # T.ToTensor() 已经将 [0, 255] 转换为 [0.0, 1.0]
        # 现在只需简单映射到 [-1, 1]
        # 【关键】移除了 robust_normalize，因为它会破坏数据分布

        pet_tensor = pet_tensor * 2.0 - 1.0
        ct_tensor = ct_tensor * 2.0 - 1.0

        return {
            'pet': pet_tensor,
            'ct': ct_tensor
        }

    # 为了兼容接口保留这些空方法
    def set_pet_max_suv(self, max_suv):
        pass

    def compute_pet_max_suv(self):
        return 1.0


class PairedDataset(Dataset):
    def __init__(self, pet_folder, ct_folder, image_size=128, augment=False, normalize=True,
                 use_medical_preprocessing=False, ct_window_width=400, ct_window_center=50,
                 pet_max_suv=None, pet_target_range="[-1,1]"):
        self.dataset = MedicalImageDataset(
            pet_folder, ct_folder, image_size, augment, normalize,
            use_medical_preprocessing, ct_window_width, ct_window_center,
            pet_max_suv, pet_target_range
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        return torch.cat([data['pet'], data['ct']], dim=0)


def get_dataloaders(config):
    if config.get('use_medical_preprocessing', False):
        raise ValueError(
            "use_medical_preprocessing is not supported by the current dataset pipeline. "
            "Set use_medical_preprocessing: false."
        )

    # Check if using same dataset for train and val
    same_paths = (
        config['train_pet_path'] == config['val_pet_path'] and
        config['train_ct_path'] == config['val_ct_path']
    )

    # 1. 创建训练集
    train_dataset = PairedDataset(
        pet_folder=config['train_pet_path'],
        ct_folder=config['train_ct_path'],
        image_size=config['image_size'],
        augment=config.get('augment', True),
        normalize=config.get('normalize', True),
        use_medical_preprocessing=config.get('use_medical_preprocessing', False),
        ct_window_width=config.get('ct_window_width', 400),
        ct_window_center=config.get('ct_window_center', 50),
        pet_max_suv=config.get('pet_max_suv', None),
        pet_target_range=config.get('pet_target_range', "[-1,1]")
    )

    # If using same paths, share the dataset instance to ensure identical data
    if same_paths:
        val_dataset = train_dataset  # Use the SAME instance!
    else:
        # 2. 创建验证集
        val_dataset = PairedDataset(
            pet_folder=config['val_pet_path'],
            ct_folder=config['val_ct_path'],
            image_size=config['image_size'],
            augment=False,
            normalize=config.get('normalize', True),
            use_medical_preprocessing=config.get('use_medical_preprocessing', False),
            ct_window_width=config.get('ct_window_width', 400),
            ct_window_center=config.get('ct_window_center', 50),
            pet_max_suv=config.get('pet_max_suv', None),
            pet_target_range=config.get('pet_target_range', "[-1,1]")
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
        shuffle=False,  # Important: Don't shuffle validation data
        num_workers=config.get('num_workers', 4),
        pin_memory=True
    )

    return train_loader, val_loader
