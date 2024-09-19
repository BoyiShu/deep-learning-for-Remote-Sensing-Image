import os, math, random, glob, time
import numpy as np
import h5py
from pyrsgis import raster
from pyrsgis.ml import imageChipsFromFile
from sklearn.model_selection import train_test_split

# Change the working directory
output_directory = r"D:\h5"
os.chdir(output_directory)

# define the file names 打开影像和真值
feature_file = r"D:\pyhsj3\9_optics_sub_8_3.tif"
label_file = r"D:\pyhsj3\9_labels_sub_8_3.tif"

# create feature chips using pyrsgis 切片
features = imageChipsFromFile(feature_file, x_size=19, y_size=19)#读取输入图片 并切片为19x19
features=features/features.max()
# read the label file and reshape it
ds, labels = raster.read(label_file)#读取特征文件并展开
labels = labels.flatten()
size,x,y,bands=features.shape
X_index = np.arange(0, size)
#Passing index array instead of the big feature matrix
X_train, X_test, train_labels, test_labels = train_test_split(X_index, labels, train_size=0.8, random_state=42)
train_data = features[X_train,:,:,:]
print("train_data.shape:",train_data.shape)
test_data = features[X_test,:,:,:]
print("test_data.shape:",test_data.shape)

# print basic details 输出大小、最大值最小值
print('Input features shape:', features.shape)
print('\nInput labels shape:', labels.shape)
# print('Values in input features, min: %d & max: %d' % (features.min(), features.max()))
print('Values in input labels, min: %d & max: %d' % (labels.min(), labels.max()))

print('Arrays saved at location %s' % (os.getcwd()))

h5f = h5py.File(r'D:\h5\pyh.h5', 'w')
h5f.create_dataset('train_x', data=train_data)
h5f.create_dataset('train_y', data=train_labels)
h5f.create_dataset('test_x', data=test_data)
h5f.create_dataset('test_y', data=test_labels)
h5f.create_dataset('features', data=features)
h5f.create_dataset('labels', data=labels)
h5f.close()
