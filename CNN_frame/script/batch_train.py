"""
可以批量训练
"""
from osgeo import gdal
import torch
from torch import nn
gdal.UseExceptions()
from PIL import Image
from tools.engine import create_model
from tools.engine import create_optimizer,predict,train
from tools import utils
from tools.utils import pretreatment
from torch.utils.data import DataLoader
import argparse
def get_args_parser():
    parser = argparse.ArgumentParser('model training and evaluation script', add_help=False)

    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')

    # Model parameters
    parser.add_argument('--model', default='resnet34', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--in_channels', default=6, type=int,
                        help='channels')
    parser.add_argument('--num_classes', default=9, type=int,
                        help='the number of classes')
    parser.add_argument('--output_dir', type=str, default='./work_dir',help='Directory to save outputs')
    parser.add_argument('--data-path', type=str, default=r"D:\pyh.h5",
                        help='data-path')
    parser.add_argument('--label-path', type=str,
                        default=r"D:\pyhsj2\9_labels_sub_5_2.tif",
                        help='label-path')

    parser.add_argument('--weight-path', type=str,
                        default=r"best_resnet_model.pth",
                        help='weight-path')
    return parser
# 配置参数
models = [
    {"name": "resnet34", "num_classes": 9, "in_channels": 6},
    # 添加其他模型配置
]
weight_paths = [
    "model1_weights.pth",
    # 添加其他模型的权重保存路径
]
data_paths = [
    "D:\pyh.h5",
    # 添加其他预测结果保存路径
]
label_paths = [
    "D:\pyhsj2\9_labels_sub_5_2.tif",
    # 添加其他预测结果保存路径
]

def batch_process(models, data_path, label_path,weight_paths,args):
    utils.init_distributed_mode(args)

    print(args)


    for i in range(len(models)):
        print(f"Processing model {i+1}/{len(models)}")
        device = torch.device(args.device)
        train_dataset, test_dataset = pretreatment(data_path[i], label_path[i])
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
        val_dataloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)

        model = create_model(models[i]['name'], models[i]['num_classes'], models[i]['in_channels'])
        model = model.to(device)
        criterion = nn.CrossEntropyLoss().to(device)
        optimizer=create_optimizer(args.opt,args.lr,model)

        weight_path = weight_paths[i]


        # 训练模型
        train(model,train_dataloader,val_dataloader,device,args.epochs,criterion,optimizer, args.batch_size,weight_path)




parser = argparse.ArgumentParser('resnet training and evaluation script', parents=[get_args_parser()])
args = parser.parse_args()
# 调用批量处理函数
batch_process(models, data_paths, label_paths,weight_paths,args)
