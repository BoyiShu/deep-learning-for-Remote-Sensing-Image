from model import fcn_resnet50,VGG16UNet
import cv2
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
def manual_preprocess(image):
    # 转换为Tensor并调整图像大小
    image = Image.open(image).convert('RGB')
    image = np.array(image, dtype=np.float32)
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts image to tensor and scales to [0, 1]
        transforms.Normalize(mean=[0.288, 0.314, 0.293], std=[0.159, 0.134, 0.125]),  # Normalize with ImageNet stats
    ])
    image = transform(image)

    return image

def draw_features(width, height, x, savename):
    tic = time.time()
    fig = plt.figure(figsize=(16, 16))
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)
    for i in range(width * height):
        plt.subplot(height, width, i + 1)
        plt.axis('off')
        img = x[0, i, :, :]
        pmin = np.min(img)
        pmax = np.max(img)
        img = ((img - pmin) / (pmax - pmin + 0.000001)) * 255  # float在[0，1]之间，转换成0-255
        img = img.astype(np.uint8)  # 转成unit8
        img = cv2.applyColorMap(img, cv2.COLORMAP_JET)  # 生成heat map
        img = img[:, :, ::-1]  # 注意cv2（BGR）和matplotlib(RGB)通道是相反的
        plt.imshow(img)
        print("{}/{}".format(i, width * height))
    fig.savefig(savename, dpi=100)
    fig.clf()
    plt.close()
    print("time:{}".format(time.time() - tic))



path=r"E:\Train\Train\Rural\img_low\0.png"
img_tensor =  manual_preprocess(path).unsqueeze(0)
print(img_tensor.shape)
model=VGG16UNet(8)
features = []

def get_features(module, input, output):
    features.append(output)
    print(output.shape)
    draw_features(16, 16, output.detach().numpy(), "feature.png")


#print(model)
model.up2.register_forward_hook(get_features)

weight_path = r"D:\fcn\best_UNET_model.pth" # 你的权重文件路径
model.load_state_dict(torch.load(weight_path))
output=model(img_tensor)
output=output["out"]
