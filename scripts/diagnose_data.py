"""
诊断数据加载问题
检查原始图像和归一化后的图像统计信息
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms as T
import glob

# 数据路径
TRAIN_PET_PATH = "Data/subset_A/part_PET/ImageSet/PNG"
VAL_PET_PATH = "Data/subset_B/part_PET/ImageSet/PNG"
TRAIN_CT_PATH = "Data/subset_A/part_CT/train_data/ImageSet/PNG"
VAL_CT_PATH = "Data/subset_B/part_CT/train_data/ImageSet/PNG"

def load_raw_image(path):
    """加载原始PNG图像并转为numpy数组"""
    img = Image.open(path).convert('L')
    arr = np.array(img)
    return arr

def analyze_folder(folder_path, name, num_samples=5):
    """分析文件夹中的图像"""
    print(f"\n{'='*60}")
    print(f"Analyzing: {name}")
    print(f"Folder: {folder_path}")
    print(f"{'='*60}")

    paths = sorted(glob.glob(os.path.join(folder_path, '*.*')))[:num_samples]
    print(f"Found {len(paths)} images (showing first {num_samples})")

    all_pixels = []
    for i, path in enumerate(paths):
        arr = load_raw_image(path)
        all_pixels.extend(arr.flatten())

        print(f"\nImage {i+1}: {os.path.basename(path)}")
        print(f"  Shape: {arr.shape}")
        print(f"  Raw pixel range: [{arr.min()}, {arr.max()}]")
        print(f"  Raw pixel mean: {arr.mean():.2f}")
        print(f"  Raw pixel std: {arr.std():.2f}")
        print(f"  Unique values: {len(np.unique(arr))}")

        # 检查是否是纯色或异常
        if arr.max() - arr.min() < 10:
            print(f"  ⚠️  WARNING: Nearly uniform image!")

    all_pixels = np.array(all_pixels)
    print(f"\n{'='*60}")
    print(f"Overall Statistics (all pixels from {num_samples} images):")
    print(f"  Min: {all_pixels.min()}")
    print(f"  Max: {all_pixels.max()}")
    print(f"  Mean: {all_pixels.mean():.2f}")
    print(f"  Median: {np.median(all_pixels):.2f}")
    print(f"  Std: {all_pixels.std():.2f}")

    # 期望值分析
    print(f"\nExpected vs Actual:")
    print(f"  Normal PET images should have:")
    print(f"    - Mean around 50-100 (dark background with bright spots)")
    print(f"    - Min = 0 (black background)")
    print(f"    - Max = 255 (brightest areas)")

    if all_pixels.mean() > 200:
        print(f"  ⚠️  CRITICAL: Mean is {all_pixels.mean():.1f} (>200)!")
        print(f"      Images are mostly WHITE - possible issues:")
        print(f"      1. Images are inverted (background=white, foreground=black)")
        print(f"      2. Images are actually masks (binary 0/255)")
        print(f"      3. Images are corrupted")

    return all_pixels

def test_normalization(folder_path, name):
    """测试当前的归一化函数"""
    print(f"\n{'='*60}")
    print(f"Testing Normalization on: {name}")
    print(f"{'='*60}")

    paths = sorted(glob.glob(os.path.join(folder_path, '*.*')))[:3]

    def robust_normalize(x):
        """当前代码使用的归一化函数"""
        x_min = x.min()
        x_max = x.max()
        if x_max - x_min < 1e-6:
            return torch.zeros_like(x)
        x_01 = (x - x_min) / (x_max - x_min)
        return 2.0 * x_01 - 1.0

    for path in paths:
        # 加载图像
        img = Image.open(path).convert('L')
        transform = T.Compose([T.Resize((128, 128)), T.ToTensor()])
        tensor = transform(img)  # [1, 128, 128], range [0, 1]

        # 归一化
        normalized = robust_normalize(tensor)

        print(f"\n{os.path.basename(path)}:")
        print(f"  Before normalization: [{tensor.min():.3f}, {tensor.max():.3f}], mean={tensor.mean():.3f}")
        print(f"  After normalization:  [{normalized.min():.3f}, {normalized.max():.3f}], mean={normalized.mean():.3f}")

def visualize_samples():
    """可视化样本图像"""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    # 训练集
    train_paths = sorted(glob.glob(os.path.join(TRAIN_PET_PATH, '*.*')))[:4]
    for i, path in enumerate(train_paths):
        ax = axes[0, i]
        img = Image.open(path).convert('L')
        ax.imshow(img, cmap='gray')
        ax.set_title(f'Train {i+1}: {os.path.basename(path)}\nmean={np.array(img).mean():.1f}')
        ax.axis('off')

    # 验证集
    val_paths = sorted(glob.glob(os.path.join(VAL_PET_PATH, '*.*')))[:4]
    for i, path in enumerate(val_paths):
        ax = axes[1, i]
        img = Image.open(path).convert('L')
        ax.imshow(img, cmap='gray')
        ax.set_title(f'Val {i+1}: {os.path.basename(path)}\nmean={np.array(img).mean():.1f}')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('data_diagnosis_visualization.png', dpi=100)
    print(f"\n✅ Saved visualization to: data_diagnosis_visualization.png")

if __name__ == "__main__":
    print("\n" + "="*60)
    print(" DATA DIAGNOSIS TOOL")
    print("="*60)

    # 分析原始图像
    analyze_folder(TRAIN_PET_PATH, "Training PET")
    analyze_folder(VAL_PET_PATH, "Validation PET")

    # 测试归一化
    test_normalization(TRAIN_PET_PATH, "Training PET")
    test_normalization(VAL_PET_PATH, "Validation PET")

    # 可视化
    visualize_samples()

    print("\n" + "="*60)
    print(" DIAGNOSIS COMPLETE")
    print("="*60)
