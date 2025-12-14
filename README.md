# PET to CT Diffusion Model

基于MSRD-Net双流架构思想的PET到CT转换扩散模型实现。

## 项目特点

- **双流PET编码器**：结合CNN局部特征提取和Transformer全局建模
- **ControlNet风格控制**：精确的空间条件控制
- **感知损失**：保持生成图像的纹理质量
- **PyTorch实现**：完全使用标准PyTorch组件

## 环境设置

### 前置要求
- Windows 11 with WSL2
- NVIDIA GPU with CUDA 12.1
- Git

### 安装步骤

1. **在WSL中创建项目目录**
```bash
cd ~
mkdir pet-ct-diffusion
cd pet-ct-diffusion
```

2. **安装Pixi（如果未安装）**
```bash
curl -fsSL https://pixi.sh/install.sh | bash
source ~/.bashrc
```

3. **创建Pixi项目**
```bash
pixi init
```

4. **将以下内容复制到 `pyproject.toml`**：
[见项目中的pyproject.toml文件]

5. **安装依赖**
```bash
pixi install
```

6. **激活环境**
```bash
pixi shell
```

## 数据准备

### 数据集结构
```
data/
├── subset_A/          # 训练集
│   ├── part_PET/
│   │   └── ImageSet/
│   │       └── PNG/
│   └── part_CT/
│       └── train_data/
│           └── ImageSet/
│               └── PNG/
└── subset_B/          # 测试集
    ├── part_PET/
    │   └── ImageSet/
    │       └── PNG/
    └── part_CT/
        └── train_data/
            └── ImageSet/
                └── PNG/
```

### 数据迁移说明
从Windows路径迁移数据到WSL：

```bash
# 在WSL中创建数据目录
mkdir -p data/{subset_A,subset_B}/{part_PET,part_CT}/{train_data/ImageSet,ImageSet}/PNG

# 从Windows复制数据（示例路径，根据实际情况调整）
cp -r /mnt/d/e_Project/子宫内膜+小目标分割/Model/My_diffusion/data/subset_A/* data/subset_A/
cp -r /mnt/d/e_Project/子宫内膜+小目标分割/Model/My_diffusion/data/subset_B/* data/subset_B/
```

## 训练

```bash
# 训练模型
pixi run python train.py
```

## 项目结构

```
.
├── models.py              # 双流编码器模型定义
├── pet_controlnet.py      # ControlNet风格的控制模块
├── dataset.py             # 数据加载器实现
├── loss.py                # 感知损失函数
├── train.py               # 训练脚本
├── test.py                # 测试脚本
├── utils.py               # 工具函数
├── pyproject.toml         # 项目配置
└── README.md              # 项目说明
```

## 模型架构

1. **双流PET编码器**
   - Local Stream: ResNet块提取局部特征
   - Global Stream: Transformer捕获全局依赖
   - 特征融合: 1x1卷积融合双流特征

2. **ControlNet注入**
   - 复制U-Net编码器结构
   - 零卷积层确保稳定训练
   - 多尺度PET特征注入

3. **损失函数**
   - 标准扩散MSE损失
   - 感知损失保持纹理质量
   - 可选的梯度惩罚