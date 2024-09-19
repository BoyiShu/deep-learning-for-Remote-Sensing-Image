# -*- coding: utf-8 -*-
import os
import numpy as np
from pyrsgis import raster
from PIL import Image
import re

# 切换工作目录
os.chdir(r"")

# 加载参考影像地理坐标
ds, featuresHyderabad = raster.read(r"")

# 指定结果保存路径
result_folder = r""

# 定义函数来识别关键词
def identify_keywords(file_name):
    keywords = re.findall(r'S12|S1|S2|SVM|RF|XGB|CNNLSVM|cnnLsvm|cnnRF|CNNRF|CNNXGB|CNNSVM|CNN|VGG|alexnet', file_name)
    return '-'.join(keywords)

# 获取文件夹中的所有.npy文件
for file_name in os.listdir(result_folder):
    if file_name.endswith(".npy"):
        # 拼接完整路径
        file_path = os.path.join(result_folder, file_name)

        # 加载预测结果
        prediction = np.load(file_path)

        # 写入预测结果和地理坐标到tif
        prediction = np.reshape(prediction, (ds.RasterYSize, ds.RasterXSize))
        raster.export(prediction, ds, filename=f"{identify_keywords(file_name)}.tif", dtype='float')

        # 上色
        img = Image.open(f"{identify_keywords(file_name)}.tif")
        img = img.convert("RGB")
        img_array = img.load()

        width, height = img.size
        for x in range(0, width):
            for y in range(0, height):
                rgb = img_array[x, y]
                if rgb == (0, 0, 0):
                    img_array[x, y] = (255, 128, 0)
                elif rgb == (1, 1, 1):
                    img_array[x, y] = (0, 191, 255)
                elif rgb == (2, 2, 2):
                    img_array[x, y] = (0, 255, 128)
                elif rgb == (3, 3, 3):
                    img_array[x, y] = (128, 0, 255)
                elif rgb == (4, 4, 4):
                    img_array[x, y] = (255, 0, 64)
                elif rgb == (5, 5, 5):
                    img_array[x, y] = (0, 64, 255)
                elif rgb == (6, 6, 6):
                    img_array[x, y] = (255, 255, 255)
                elif rgb == (7, 7, 7):
                    img_array[x, y] = (255, 128, 128)
                elif rgb == (8, 8, 8):
                    img_array[x, y] = (128, 255, 128)
                elif rgb == (9, 9, 9):
                    img_array[x, y] = (128, 128, 255)

        img.save(os.path.join(result_folder, f"{identify_keywords(file_name)}.tif"))
