# deep-learning-for-Remote-Sensing-Image
# 目录

- [背景介绍](#背景介绍)
- [项目结构](#项目结构)
  - [CNN 框架](#cnn-框架)
  - [FCN 框架](#fcn-框架)
- [提供模型](#提供模型)
- [环境依赖](#环境依赖)
- [快速开始](#快速开始)
  - [数据处理](#数据处理)
  - [模型训练](#模型训练)
  - [模型评估](#模型评估)
  - [可视化](#可视化)
- [感谢](#感谢)
- [许可证](#许可证)
- [合作交流](#合作交流)

## 背景介绍

`deep-learning-for-Remote-Sensing-Image` 是一个专注于使用深度学习技术来进行遥感图像土地分类（classification）和分割（Segmentation）的项目。本仓库提供了两个主要的框架及其相应的代码实现：

1. **基于Patch块的卷积神经网络（CNN）框架**：该方法将遥感图像逐像素为中心划分为多个小的Patch块，基于Patch块提取特征并分类。此方法适用于处理高分辨率遥感图像，能够有效捕捉局部特征。

2. **全卷积网络（FCN）框架**：该方法直接对整幅遥感图像进行像素级别的分割，无需划分为Patch块。FCN能够生成与输入图像相同尺寸的输出分割图，适合大规模遥感图像分割任务。

## 项目结构

### CNN 框架
```markdown
- CNN_frame/
  - data/  数据处理模块
    - create_patch.py: 数据patch处理
    - disjoint_joint.py: 数据量过大可先进行数据切割后patch处理
  - evaluate/  预测与精度评估
    - colorit.py: 结果可视化工具
    - martix-class.py: 类别精度评价
    - predict.py: 预测脚本
  - model/  模型调用模块（可自行添加模型）
    - alexnet_model.py: AlexNet模型
    - convnext_model.py: ConvNext模型
    - densenet_model.py: DenseNet模型
    - efficientnetV2_model.py: EfficientNet V2模型
    - regnet_model.py: RegNet模型
    - resnet_model.py: ResNet模型
    - shufflenet_model.py: ShuffleNet模型
    - vgg_model.py: VGG模型
  - script/  训练脚本
    - batch-train.py 批量训练脚本
    - train.py: 单次训练脚本
  - tools/  工具包模块
    - engine.py: 初始化搭建模型、训练辅助
    - utils.py: 工具数据加载、分割工具
  - visualization/  可视化模块
    - feature.jpg: 特征图可视化结果示例
    - umap.jpg: UMAP降维结果示例
    - umap.py: UMAP算法
    - visualization_feature_map.py: 特征图可视化脚本
```
### FCN 框架
```markdown
- FCN_frame/
  - data/  数据处理模块
    - pic_pre.py: 图像预处理工具（分辨率/通道处理）
  - evaluate/  预测与精度评估
    - colorit.py: 结果可视化工具
    - martix-class.py: 类别精度评价
    - predict.py: 预测脚本
  - model/  模型调用模块（可自行添加模型）
    - backbone/
      - mobilenet_backbone.py
      - resnet_backbone.py
    - deeplabv3_model.py: DeepLabV3模型
    - fcn_model.py: FCN模型
    - mobilenet_unet.py: MobileNet-Unet模型
    - unet.py: UNet模型
    - vgg_unet.py: VGG-Unet模型
  - script/  训练脚本
    - train.py: 训练脚本
  - tools/  工具包模块
    - engine.py: 初始化搭建模型、训练辅助工具
    - utils.py: 数据加载、分割工具
  - visualization/  可视化模块
    - feature.png: 特征图可视化结果示例
    - umap.jpg: UMAP降维结果示例
    - umap.py: UMAP算法
    - visualization_feature_map.py: 特征图可视化脚本
```
## 提供模型

| 模型名称                          | 论文链接       |
|------------------------------------|------------|
| **FCN**                            |            |
| fcn_resnet50                       |            |
| vgg16unet                          |            |
| deeplabv3_resnet50                 |            |
| **ResNet**                         |            |
| resnet34                           |            |
| resnet50                           |            |
| resnet101                          |            |
| resnext50_32x4d                    |            |
| resnext101_32x8d                   |            |
| **AlexNet**                        |            |
| AlexNet                            |            |
| **VGG**                            |            |
| vgg11                              |            |
| vgg13                              |            |
| vgg16                              |            |
| vgg19                              |            |
| **RegNet**                         |            |
| regnet                             |            |
| **ConvNeXt**                       |            |
| convnext_tiny                      |            |
| convnext_small                     |            |
| convnext_base                      |            |
| convnext_large                     |            |
| convnext_xlarge                    |            |
| **EfficientNetV2**                 |            |
| efficientnetv2_s                   |            |
| efficientnetv2_m                   |            |
| efficientnetv2_l                   |            |
| **ShuffleNet**                     |            |
| shufflenet_v2_x0_5                 |            |
| shufflenet_v2_x1_0                 |            |
| shufflenet_v2_x1_5                 |            |
| shufflenet_v2_x2_0                 |            |
| **DenseNet**                       |            |
| densenet121                        |            |
| densenet161                        |            |
| densenet169                        |            |
| densenet201                        |            |

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


### 数据处理

1. 请参考 `README.md` 中的说明，下载并预处理遥感图像数据集。本项目提供的数据可见（链接）。将数据集放置在 `CNN_frame/data/` 或 `FCN_frame/data/` 目录下。

2. 使用CNN框架，打开`CNN_frame/data/`运行 `create_patch.py` 进行数据 patch 处理，将数据存储在h5文件中，并已划分好训练和测试集。如果数据量过大，可使用 `disjoint_joint.py` 先进行切割再进行patch处理。

3. 使用FCN框架，如果需要进行将分辨率处理或更改图片通道则打开`FCN_frame/data/`运行`pic_pre.py`进行处理。若无需处理可直接进入模型训练，详细见下文。


### 模型训练

1. 打开`CNN_frame/script`或`FCN_frame/script`。

2. 运行`train.py`，设置所需模型和基本参数即可开始训练，训练完成后自动保存模型训练文件以便预测。

3. CNN框架提供了批量训练文件`CNN_frame/script/batch_train.py`。

### 模型评估

1. 打开`CNN_frame/evaluate`或`FCN_frame/evaluate`

2. 运行预测脚本`predict.py`将输出测试集预测精度。

3. 运行 `martix-class.py`进行类别精度评价并保存为excel文件。

4. 运行`colorit.py`可自定义输出分类结果上色图。

### 可视化

1. 打开`CNN_frame/visualization`或`FCN_frame/visualization`

2. 运行 `visualization_feature_map.py` 进行特征图可视化：

## 感谢

该项目灵感来源于 **[deep-learning-for-image-processing](https://github.com/WZMIAOMIAO/deep-learning-for-image-processing)**

## 许可证

本项目使用 [MIT License](LICENSE) 进行许可。

## 合作交流
项目正在持续建设中，如果您有兴趣欢迎联系与加入我们！   
Feiya Shu：shufeiya@gmail.com   
Qinxin Wu: 2410766684@qq.com   
同时，欢迎贡献代码和提出问题！如果您有任何建议或发现了问题，请提交一个Issue或Pull Request。
