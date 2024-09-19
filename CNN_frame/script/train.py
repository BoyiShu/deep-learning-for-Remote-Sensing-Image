from torch.utils.data import DataLoader
import argparse
import torch
from torch import nn
from tools.utils import pretreatment
from tools.engine import train
from pathlib import Path
from tools import utils
from tools.engine import create_model
from tools.engine import create_optimizer
"""
可供选择的模型：resnet34 resnet50 resnet101 resnext50_32x4d resnext101_32x8d
             AlexNet
             vgg11 vgg13 vgg16 vgg19 
             regnet 
             convnext_tiny convnext_small convnext_base convnext_large convnext_xlarge
             efficientnetv2_s efficientnetv2_m efficientnetv2_l 
             shufflenet_v2_x0_5 shufflenet_v2_x1_0 shufflenet_v2_x1_5 shufflenet_v2_x2_0
             densenet121 densenet161 densenet169 densenet201
"""
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
    parser.add_argument('--data-path', type=str, default=r"D:\h5\pyh.h5",
                        help='data-path_h5')

    parser.add_argument('--weight-path', type=str,
                        default=r"best_resnet_model.pth",
                        help='weight-path')
    return parser


def main(args):
    utils.init_distributed_mode(args)

    print(args)

    device = torch.device(args.device)
    train_dataset,test_dataset=pretreatment(args.data_path)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)

    model=create_model(args.model,num_classes=args.num_classes,in_channels=args.in_channels).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer=create_optimizer(args.opt,args.lr,model)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # 将参数数量转换为百万
    params_in_million = total_params / 1e6
    print(f"Parameters(M): {params_in_million:.2f}")
    train(model,train_dataloader,val_dataloader,device,args.epochs,criterion,optimizer, args.batch_size,args.weight_path)






if __name__ == '__main__':
    parser = argparse.ArgumentParser('resnet training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
