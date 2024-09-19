import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from tools.utils import SegmentationDataset
import time
torch.backends.cudnn.enabled = False
from tools.engine import create_model
import torchvision.transforms as transforms
from umap import use
import functools

def get_args_parser():
    parser = argparse.ArgumentParser('model training and evaluation script', add_help=False)

    parser.add_argument('--batch-size', default=2, type=int)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument("--aux", default=False, type=bool, help="auxilier loss")
    # Model parameters
    parser.add_argument('--model', default='vgg16unet', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')
    parser.add_argument('--in_channels', default=3, type=int,
                        help='channels')
    parser.add_argument('--num_classes', default=8, type=int,
                        help='the number of classes')
    parser.add_argument('--output_dir', type=str, default='./work_dir', help='Directory to save outputs')
    parser.add_argument('--data-path', type=str, default=r"E:\Train\Train\Rural\1",
                        help='data-path')
    parser.add_argument('--label-path', type=str,
                        default=r"E:\Train\Train\Rural\2",
                        help='label-path')
    parser.add_argument('--featuretxt-path', type=str,
                        default=r"features.txt",
                        help='feature-path')
    parser.add_argument('--labeltxt-path', type=str,
                        default=r"labels.txt",
                        help='save-path')
    parser.add_argument('--weight-path', type=str,
                        default=r"D:\fcn\fcn\best_UNET_model_pretrained.pth",
                        help='weight-path')
    parser.add_argument('--savepicture-path', type=str,
                        default=r"umap.jpg",
                        help='save-path')
    return parser

def hook_fn(args, module, input, output):
    global feature_maps
    # 获取特征图

def predict(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size
    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    # 加载数据
    image_folder = args.data_path
    label_folder = args.label_path
    image_paths = [os.path.join(image_folder, fname) for fname in os.listdir(image_folder)]
    label_paths = [os.path.join(label_folder, fname) for fname in os.listdir(label_folder)]
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts image to tensor and scales to [0, 1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize with ImageNet stats
    ])
    dataset = SegmentationDataset(image_paths, label_paths, transform)
    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             shuffle=False,
                             pin_memory=True,
                             collate_fn=None)  # 如果不需要特殊的collate_fn，可以设为None
    # 创建模型
    model = create_model(aux=args.aux, model_name=args.model, num_classes=args.num_classes)
    weights_dict = torch.load(args.weight_path, map_location='cpu')
    for k in list(weights_dict.keys()):
        if "aux" in k:
            del weights_dict[k]

    model.to(device)
    model.load_state_dict(weights_dict)
    handle = model.up2.register_forward_hook(functools.partial(hook_fn, args))
    model.eval()
    predictions = []
    labels_list = []
    with torch.no_grad():
        for i ,(inputs, labels) in enumerate(data_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            output = model(inputs)["out"]
            print(output.shape)
            feature = output.cpu().numpy()
            with open(args.featuretxt_path, "a") as f:
                for b in range(feature.shape[0]):  # 遍历批次
                    for h in range(feature.shape[2]):  # 遍历高度
                        for w in range(feature.shape[3]):  # 遍历宽度
                            # 提取像素(h, w)的所有通道值
                            pixel_values = feature[b, :, h, w]
                            # 将像素值转换为字符串并用空格隔开
                            pixel_str = " ".join(str(x) for x in pixel_values)
                            # 写入文件
                            f.write(pixel_str + "\n")
            print(labels.shape)
            labels_list.extend(labels.cpu().numpy())
    labels = np.concatenate(labels_list)
    labels_flattened = labels.flatten()

        # 将每个像素值写入到文本文件中
    np.savetxt(args.labeltxt_path, labels_flattened, fmt='%d')
def use_umap(args):
    use(args.featuretxt_path,args.labeltxt_path,args.savepicture_path)
if __name__ == '__main__':
    # check GPU
    parser = argparse.ArgumentParser('model prediction script', parents=[get_args_parser()])
    args = parser.parse_args()
    start_time = time.time()
    predict(args)
    end_time = time.time()
    print('Finished predict')
    spend_train = end_time - start_time  # 计算时间差，得出的结果即为程序运行时间
    print("执行了{}秒".format(spend_train))
    use_umap(args)