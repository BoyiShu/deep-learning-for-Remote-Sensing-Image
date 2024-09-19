import torchvision
import  random
from matplotlib import pyplot as plt
from model import *
import cv2
import  torch
from osgeo import gdal
import numpy as np
import torchvision.transforms as transforms
def manual_preprocess(image):
    # 转换为Tensor并调整图像大小
    image = torch.tensor(image, dtype=torch.float32) / 255.0  # 将像素值缩放到 [0, 1]
    image = image.permute(2, 0, 1)  # 转换为 (C, H, W) 格式

    # 计算均值和标准差
    mean = image.mean(dim=(1, 2), keepdim=True)
    std = image.std(dim=(1, 2), keepdim=True)

    # 标准化
    image = (image - mean) / std

    return image


def random_num(size, end):
    range_ls = [i for i in range(end)]
    num_ls = []
    for i in range(size):
        num = random.choice(range_ls)
        range_ls.remove(num)
        num_ls.append(num)
    return num_ls
path="E:\pyhsj2\9_optics_sub_5_2.tif"
dataset = gdal.Open(path)
img = dataset.ReadAsArray()
img = np.transpose(img, (1, 2, 0))
img_tensor = manual_preprocess(img).unsqueeze(0)

model = resnet34(num_classes=9,in_channels=6)
weight_path = r"D:\main\best_resnet_model.pth"  # 你的权重文件路径
#model.load_state_dict(torch.load(weight_path))
model.eval()
print(model)
new_model = torchvision.models._utils.IntermediateLayerGetter(model, {'layer1': '1', 'layer2': '2', 'layer3': '3'})
out = new_model(img_tensor)

tensor_ls = [(k, v) for k, v in out.items()]

# 选取conv2的输出
v = tensor_ls[1][1]

# 取消Tensor的梯度并转成三维tensor，否则无法绘图
v = v.data.squeeze(0)

print(v.shape)  # torch.Size([512, 28, 28])

# 随机选取25个通道的特征图
channel_num = random_num(25, v.shape[0])
plt.figure(figsize=(10, 10))
for index, channel in enumerate(channel_num):
    ax = plt.subplot(5, 5, index + 1, )
    plt.imshow(v[channel, :, :])  # 灰度图参数cmap="gray"
plt.savefig("feature2.jpg", dpi=300)


