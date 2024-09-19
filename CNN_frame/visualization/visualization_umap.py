import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
from torch.utils.data import DataLoader, TensorDataset
from osgeo import gdal
gdal.UseExceptions()
import numpy as np
import torch
import time
from tools import utils
from tools.utils import pretreatment_predict
from tools.engine import create_model
from umap import use
import argparse
import functools


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

    parser.add_argument('--data-path', type=str, default=r"D:\pyh.h5",
                        help='data-path')
    parser.add_argument('--label-path', type=str,
                        default=r"D:\pyhsj2\9_labels_sub_5_2.tif",
                        help='label-path')

    parser.add_argument('--weight-path', type=str,
                        default=r"best_resnet_model.pth",
                        help='weight-path')

    parser.add_argument('--featuretxt-path', type=str,
                        default=r"features.txt",
                        help='feature-path')
    parser.add_argument('--labeltxt-path', type=str,
                        default=r"labels.txt",
                        help='save-path')
    parser.add_argument('--savepicture-path', type=str,
                        default=r"umap.jpg",
                        help='save-path')

    return parser
def hook_fn(args, module, input, output):
    global feature_maps
    # 获取特征图并将其展平成一维数组
    feature = output.cpu().numpy()
    feature = feature.reshape(feature.shape[0], -1)
    with open(args.featuretxt_path, "a") as f:
        for i in range(feature.shape[0]):
            # 将一组512个数字写入文件，相邻数字用空格隔开
            f.write(" ".join(str(x) for x in feature[i]))
            # 写入换行符，分隔不同的组
            f.write("\n")
def predict(args):

    utils.init_distributed_mode(args)

    print(args)

    device = torch.device(args.device)
    full_dataset = pretreatment_predict(args.data_path, args.label_path)
    full_loader = DataLoader(dataset=full_dataset, batch_size=args.batch_size, shuffle=False)
    model = create_model(args.model, num_classes=args.num_classes, in_channels=args.in_channels).to(device)
    model.load_state_dict(torch.load(args.weight_path))
    handle = model.fc.register_forward_hook(functools.partial(hook_fn, args))
    model.eval()
    predictions = []
    labels_list = []
    with torch.no_grad():
        # 取前64个样本
        for i, (inputs, labels) in enumerate(full_loader):
            if i == 64:  # 仅取前64个样本
                break
            # inputs = F.pad(inputs, (0, 0, 6, 7, 6, 7), mode='constant', value=0)
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.permute(0, 3, 1, 2)
            outputs = model(inputs)
            predicted = torch.max(outputs, dim=1)[1]
            predictions.extend(predicted.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())
        # 改变数值类型
        y_pred_img = np.array(predictions).astype(np.uint8)
        # 将 labels 写入到 txt 文件中
        labels = np.array(labels_list)
        np.savetxt(args.labeltxt_path, labels, fmt='%d')

    return y_pred_img
def use_umap(args):
    use(args.featuretxt_path,args.labeltxt_path,args.savepicture_path)

if __name__ == '__main__':
    # check GPU
    parser = argparse.ArgumentParser('model predict script', parents=[get_args_parser()])
    args = parser.parse_args()
    start_time = time.time()
    predict(args)
    end_time = time.time()
    print('Finished predict')
    spend_train = end_time - start_time  # 计算时间差，得出的结果即为程序运行时间
    print("执行了{}秒".format(spend_train))
    use_umap(args)

