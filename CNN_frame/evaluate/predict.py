import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
from torch.utils.data import DataLoader, TensorDataset

from PIL import Image
from osgeo import gdal
gdal.UseExceptions()
import argparse
import numpy as np
import torch
from pathlib import Path
from tools import utils
import time
import torch.nn.functional as F
from tools.utils import load_data, preprocess_data
from tools.utils import pretreatment_predict,pretreatment_predict_img
from tools.engine import create_model
from torch import nn
from tools.utils import calculate_confusion_matrix,calculate_kappa,calculate_oa, calculate_AA,calculate_PA,calculate_UA
def get_args_parser():
    parser = argparse.ArgumentParser('model predict script', add_help=False)

    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')

    # Model parameters
    parser.add_argument('--model', default='resnet34', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--in_channels', default=6, type=int,
                        help='channels')
    parser.add_argument('--num_classes', default=9, type=int,
                        help='the number of classes')

    parser.add_argument('--data-path', type=str, default=r"D:\h5\pyh.h5",
                        help='data-path')
    parser.add_argument('--label-path', type=str, default=r"D:\pyhsj3\9_labels_sub_8_3.tif",
                        help='label-path')
    parser.add_argument('--weight-path', type=str,
                        default=r"D:\CNN_frame\script\best_resnet_model.pth",
                        help='weight-path')

    parser.add_argument('--save-path', type=str,
                        default=r"pyh.tif",
                        help='save-path')

    return parser
def predict(args):
    utils.init_distributed_mode(args)

    print(args)

    device = torch.device(args.device)
    full_dataset=pretreatment_predict(args.data_path)
    full_loader = DataLoader(dataset=full_dataset, batch_size=args.batch_size, shuffle=False)

    model = create_model(args.model, num_classes=args.num_classes, in_channels=args.in_channels).to(device)
    model.load_state_dict(torch.load(args.weight_path))

    model.eval()
    predictions = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in full_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.permute(0, 3, 1, 2)
            outputs = model(inputs)
            predicted = torch.max(outputs, dim=1)[1]
            predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    matrix = calculate_confusion_matrix(all_labels, predictions)
    OA = calculate_oa(matrix)
    kappa = calculate_kappa(OA, matrix)
    AA = calculate_AA(matrix)
    PA= calculate_PA(matrix)
    UA = calculate_UA(matrix)
    print('Finished Accuracy evaluation')
    print("OA", OA)
    print("kappa", kappa)
    print("AA", AA)
    print("PA", PA)
    print("UA", UA)

def predict_img(args):
    utils.init_distributed_mode(args)

    print(args)

    device = torch.device(args.device)
    full_dataset=pretreatment_predict_img(args.data_path)
    full_loader = DataLoader(dataset=full_dataset, batch_size=args.batch_size, shuffle=False)

    model = create_model(args.model, num_classes=args.num_classes, in_channels=args.in_channels).to(device)
    model.load_state_dict(torch.load(args.weight_path))

    model.eval()
    predictions = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in full_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.permute(0, 3, 1, 2)
            outputs = model(inputs)
            predicted = torch.max(outputs, dim=1)[1]
            predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())


    # 改变数值类型
    y_pred_img = np.array(predictions).astype(np.uint8)

    # 将预测结果转换为图像形状并且保存
    tag_g_ds = gdal.Open(args.label_path)
    tag_g_data = tag_g_ds.ReadAsArray()
    print("tag_g_data", tag_g_data.shape)
    height, width = tag_g_data.shape
    y_pred_img = y_pred_img.reshape(height, width)

    image = Image.fromarray(y_pred_img)
    image.save(args.save_path)

    return y_pred_img








if __name__ == '__main__':
    parser = argparse.ArgumentParser('model predict script', parents=[get_args_parser()])
    args = parser.parse_args()
    start_time = time.time()
    predict(args)
    predict_img(args)
    end_time = time.time()
    print('Finished predict')
    spend_train = end_time - start_time  # 计算时间差，得出的结果即为程序运行时间
    print("执行了{}秒".format(spend_train))