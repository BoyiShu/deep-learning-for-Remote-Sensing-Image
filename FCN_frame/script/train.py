import os
import argparse
import torchvision.transforms as transforms
import torch
from torch.utils.data import  DataLoader
from tools.utils import SegmentationDataset
torch.backends.cudnn.enabled = False
from sklearn.model_selection import train_test_split
from tools.engine import create_model,create_optimizer,train,create_lr_scheduler

"""
可供选择的模型：fcn_resnet50 vgg16unet deeplabv3_resnet50
"""
def get_args_parser():
    parser = argparse.ArgumentParser('training and evaluation script', add_help=False)

    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')

    # Model parameters
    parser.add_argument('--model', default='fcn50', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument("--aux", default=True, type=bool, help="auxilier loss")
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--in_channels', default=3, type=int,
                        help='channels')
    parser.add_argument('--num_classes', default=8, type=int,
                        help='the number of classes')
    parser.add_argument('--data-path', type=str, default=r"J:\Train\Train\Rural\img_low",
                        help='data-path')
    parser.add_argument('--label-path', type=str,
                        default=r"J:\Train\Train\Rural\label_new",
                        help='label-path')

    parser.add_argument('--weight-path', type=str,
                        default=r"best_UNET_model_pretrained3.pth",
                        help='weight-path')
    parser.add_argument("--amp", default=False, type=bool,
                        help="Use torch.cuda.amp for mixed precision training")
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    return parser


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    # 用来保存训练以及验证过程中信息

    batch_size = args.batch_size
    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    #加载数据
    image_folder = args.data_path
    label_folder = args.label_path
    image_paths = [os.path.join(image_folder, fname) for fname in os.listdir(image_folder)]
    label_paths = [os.path.join(label_folder, fname) for fname in os.listdir(label_folder)]
    image_paths.sort()
    label_paths.sort()
    train_image_paths, test_image_paths, train_label_paths, test_label_paths = train_test_split(
        image_paths, label_paths, test_size=0.2, random_state=42
    )
    print(f"训练集大小: {len(train_image_paths)}")
    print(f"测试集大小: {len(test_image_paths)}")
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts image to tensor and scales to [0, 1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize with ImageNet stats
    ])

    train_dataset = SegmentationDataset(train_image_paths ,train_label_paths,transform)
    val_dataset = SegmentationDataset(test_image_paths, test_label_paths,transform)
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              shuffle=True,
                              pin_memory=True,
                              collate_fn=None)  # 如果不需要特殊的collate_fn，可以设为None
    val_loader = DataLoader(val_dataset,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              shuffle=False,
                              pin_memory=True,
                              collate_fn=None)  # 如果不需要特殊的collate_fn，可以设为None
    #创建模型
    model = create_model(aux=args.aux, num_classes=args.num_classes,model_name=args.model)
    model.to(device)
    for images, labels in train_loader:
        print(f"Images shape: {images.shape}")  # Should be [batch_size, 8, H, W]
        print(f"Labels shape: {labels.shape}")  # Should be [batch_size, H, W]
        break  # Just print one batch
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # 将参数数量转换为百万
    params_in_million = total_params / 1e6
    print(f"Parameters(M): {params_in_million:.2f}")
    optimizer = create_optimizer(args.opt, args.lr, model)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True)
    train(args.epochs,train_loader,val_loader,optimizer,model,args.aux,args.weight_path,device,args.batch_size,args.num_classes,lr_scheduler)







if __name__ == '__main__':
    parser = argparse.ArgumentParser('training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
