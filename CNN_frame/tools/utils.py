import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
from torch.utils.data import DataLoader, TensorDataset

from PIL import Image
from osgeo import gdal
import rasterio
gdal.UseExceptions()

import numpy as np
import torch
import h5py
from collections import Counter


def split_dataset(features_tensor, labels_tensor, train_size=0.05, random_state=42):
    """
    划分训练集和测试集
    """
    # print("features_tensor, labels_tensor:", features_tensor.shape, labels_tensor.shape)
    X_train, X_test, y_train, y_test = train_test_split(
        features_tensor, labels_tensor, train_size=train_size, random_state=random_state
    )
    print('Reshaped features, train_x.shape, test_x.shape:', X_train.shape, X_test.shape)
    Samples, rowSize, colSize, nBands = X_train.shape  # 得到最终训练集的长度
    print("得到最终训练集的形状")
    print("Samples:", Samples, "rowSize:", rowSize, "colSize:", colSize, "nBands:", nBands)
    # print("X_train, X_test:", X_train.shape, X_test.shape)
    return X_train, X_test, y_train, y_test


def create_dataloaders(X_train, y_train, X_test, y_test, batch_size=64):
    """
    创建数据加载器
    """
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def load_data(data_path):
    """
    载入特征和标签数据
    """
    h5f = h5py.File(data_path, 'r')
    features = np.asarray(h5f['features'])
    labels = np.asarray(h5f['labels'])

    # 检查数据类型并转换
    if features.dtype != np.float32:
        features = features.astype(np.float32)
    if labels.dtype != np.int64:
        labels = labels.astype(np.int64)

    # 统计标签类别数量
    label_counts = Counter(labels)
    print("Label Counts:")
    for label, count in label_counts.items():
        print(f"Label {label}: {count} samples")

    return features, labels




def load_data_v2(data_path):
    """
    Load data from HDF5 file and preprocess it.

    Args:
        data_path (str): Path to the HDF5 file containing the data.

    Returns:
        torch.Tensor: Preprocessed data tensor.
    """
    # Load data from HDF5 file
    with h5py.File(data_path, 'r') as f:
        data = f['data'][:]

    # Normalize data
    data = (data - data.min()) / (data.max() - data.min())

    # Convert to PyTorch tensor
    data = torch.from_numpy(data).uint8()

    return data





def load_data_v3(data_path):
    """
    载入特征和标签数据
    """
    h5f = h5py.File(data_path, 'r')
    train_x = np.asarray(h5f['train_x'])
    train_y = np.asarray(h5f['train_y'])
    test_x  = np.asarray(h5f['test_x'])
    test_y = np.asarray(h5f['test_y'])
    features = np.asarray(h5f['features'])
    labels = np.asarray(h5f['labels'])



    # 检查数据类型并转换
    if train_x.dtype != np.float32:
        train_x = train_x.astype(np.float32)
    if test_x.dtype != np.float32:
        test_x = test_x.astype(np.float32)
    if features.dtype != np.float32:
        features = features.astype(np.float32)

    return train_x,train_y,test_x,test_y,features,labels





def preprocess_data_v2(data_generator, labels, batch_size):
    """
    Preprocess data and labels for training.
    """
    features = []
    for batch in data_generator:
        features.append(batch)
        if len(features) == batch_size:
            features_tensor = torch.Tensor(features)
            labels_tensor = torch.Tensor(labels).long()
            yield features_tensor, labels_tensor
            features = []
    # Process any remaining data
    if features:
        features_tensor = torch.Tensor(features)
        labels_tensor = torch.Tensor(labels).long()
        yield features_tensor, labels_tensor


import torch.utils.data as data


def create_dataloaders_v2(X_train, y_train, X_test, y_test, batch_size):
    """
    Create PyTorch DataLoader objects for training and testing.
    """
    train_dataset = data.TensorDataset(X_train, y_train)
    test_dataset = data.TensorDataset(X_test, y_test)

    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader




def preprocess_data(train_x,train_y,test_x,test_y,features,labels):
    """
    对特征数据进行标准化
    """


    train_features_tensor = torch.Tensor(train_x)

    test_features_tensor = torch.Tensor(test_x)
    features_tensor = torch.Tensor(features)

    train_labels_tensor = torch.LongTensor(train_y.astype(np.int64))
    test_labels_tensor = torch.LongTensor(test_y.astype(np.int64))
    labels_tensor = torch.LongTensor(labels.astype(np.int64))

    mean = train_features_tensor.mean(dim=(0, 1, 2))
    std = train_features_tensor.std(dim=(0, 1, 2))
    train_features_tensor = (train_features_tensor - mean) / std
    test_features_tensor = (test_features_tensor - mean) / std
    features_tensor = (features_tensor - mean) / std



    return train_features_tensor, train_labels_tensor,test_features_tensor,test_labels_tensor,features_tensor,labels_tensor

def calculate_oa(cm):
    total_correct = np.trace(cm)
    total_samples = np.sum(cm)
    oa = total_correct / total_samples
    return oa


def calculate_kappa(oa, cm):
    total_samples = np.sum(cm)
    pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / (total_samples ** 2)
    kappa = (oa - pe) / (1 - pe)
    return kappa

def calculate_AA(cm):
    PA = np.diag(cm) / np.sum(cm, axis=1)  # 计算每一类的PA
    AA = np.mean(PA)  # 计算所有类的平均PA（即AA）
    return AA

def calculate_PA(cm):
    PA = np.diag(cm) / np.sum(cm, axis=1)  # Producer's Accuracy

    return PA
def calculate_UA(cm):
    UA = np.diag(cm) / np.sum(cm, axis=0)  # User's Accuracy
    return UA

def calculate_confusion_matrix(y_true, y_pred):
    """
    计算混淆矩阵

    Args:
        y_true (list or numpy.ndarray): 真实标签
        y_pred (list or numpy.ndarray): 预测标签

    Returns:
        numpy.ndarray: 混淆矩阵
    """
    return confusion_matrix(y_true, y_pred)

def print_class_distribution(y, dataset_name):
    class_counts = Counter(y.numpy())
    total_samples = len(y)
    print(f"\n{dataset_name} 类别分布:")
    for class_label, count in sorted(class_counts.items()):
        percentage = (count / total_samples) * 100
        print(f"类别 {class_label}: {count} 样本 ({percentage:.2f}%)")

def pretreatment(data_path):
    train_x,train_y,test_x,test_y,all_x,all_y=load_data_v3(data_path)
    print("features", all_x.shape)
    train_y[train_y == 255] = 8
    test_y[test_y == 255] = 8
    all_y[all_y == 255] = 8
    # 统计标签类别数量
    label_counts = Counter(all_y)
    print("Label Counts:")
    for label, count in label_counts.items():
        print(f"Label {label}: {count} samples")

    X_train, y_train,X_test,y_test,data_tensor,labels = preprocess_data(train_x,train_y,test_x,test_y,all_x,all_y)
    print("成功预处理")


    total_samples = len(data_tensor)
    print(f"总样本量: {total_samples}")

    # 输出每个数据集的样本量
    print(f"训练集样本量: {len(X_train)} ({len(X_train) / total_samples * 100:.2f}%)")
    print(f"验证集样本量: {len(X_test)} ({len(X_test) / total_samples * 100:.2f}%)")

    # 输出每个数据集中每一类的样本量
    print_class_distribution(y_train, "训练集")
    print_class_distribution(y_test, "测试集")
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    return train_dataset,test_dataset


def pretreatment_predict_img(data_path):
    train_x,train_y,test_x,test_y,features,labels=load_data_v3(data_path)
    print("features ", features.shape)
    labels[labels == 255] = 8
    print("labels", labels.shape)

    X_train, y_train,X_test,y_test,features_tensor, labels_tensor = preprocess_data(train_x,train_y,test_x,test_y,features, labels)
    full_dataset = TensorDataset(features_tensor, labels_tensor)
    return full_dataset
def pretreatment_predict(data_path):
    train_x,train_y,test_x,test_y,features,labels=load_data_v3(data_path)
    print("data", test_x.shape)
    test_y[test_y == 255] = 8
    print("labels", test_y.shape)

    X_train, y_train,X_test,y_test,features_tensor, labels_tensor = preprocess_data(train_x,train_y,test_x,test_y,features, labels)
    val_dataset = TensorDataset(X_test, y_test)
    return val_dataset


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print
def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


