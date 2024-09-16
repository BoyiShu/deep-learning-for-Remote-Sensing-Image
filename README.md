# deep-learning-for-Remote-Sensing-Image
```markdown
# deep-learning-for-Remote-Sensing-Image

## Introduction

`deep-learning-for-Remote-Sensing-Image` 是一个专注于使用深度学习技术来进行遥感图像土地分割和分类的项目。本仓库提供了两个主要的框架及其相应的代码实现：

1. **基于Patch块的卷积神经网络（CNN）框架**：该方法将遥感图像划分为多个小的Patch块，然后使用CNN对这些Patch块进行分类。此方法适用于处理高分辨率遥感图像，能够有效捕捉局部特征。

2. **全卷积网络（FCN）框架**：该方法直接对整幅遥感图像进行像素级别的分割，无需划分为Patch块。FCN能够生成与输入图像相同尺寸的输出分割图，适合大规模遥感图像分割任务。

## 项目结构

```
deep-learning-for-Remote-Sensing-Image/
│
├── Patch-CNN/
│   ├── models/               # Patch-CNN 模型定义
│   ├── data/                 # 数据处理与加载脚本
│   ├── train.py              # 模型训练脚本
│   └── evaluate.py           # 模型评估脚本
│
├── FCN/
│   ├── models/               # FCN 模型定义
│   ├── data/                 # 数据处理与加载脚本
│   ├── train.py              # 模型训练脚本
│   └── evaluate.py           # 模型评估脚本
│
├── datasets/                 # 遥感图像数据集相关信息
│   ├── common_dataset/       # 数据处理与加载脚本
│   ├── our_dataset/       # 数据处理与加载脚本
│   └── README.md             # 数据集介绍及下载链接
│
└── README.md                 # 项目介绍
```

## 环境依赖

要运行本项目，请确保您的环境中安装了以下软件包：

- Python 3.8+
- PyTorch 1.9+
- NumPy
- OpenCV
- scikit-learn

可以通过以下命令安装依赖：

```bash
pip install -r requirements.txt
```

## 快速开始

### 1. 准备数据

请参考 `datasets/README.md` 中的说明，下载并预处理遥感图像数据集。将处理后的数据集放置在 `Patch-CNN/data/` 或 `FCN/data/` 目录下。

### 2. 训练模型

#### Patch-CNN

```bash
cd Patch-CNN
python train.py --config config.yaml
```

#### FCN

```bash
cd FCN
python train.py --config config.yaml
```

### 3. 评估模型

训练完成后，可以运行以下命令对模型进行评估：

#### Patch-CNN

```bash
cd Patch-CNN
python evaluate.py --model_path /path/to/your/model.pth
```

#### FCN

```bash
cd FCN
python evaluate.py --model_path /path/to/your/model.pth
```

## 贡献

欢迎贡献代码和提出问题！如果您有任何建议或发现了问题，请提交一个Issue或Pull Request。

## 许可证

本项目使用 [MIT License](LICENSE) 进行许可。
```
