import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from tools.utils import SegmentationDataset
from PIL import Image
torch.backends.cudnn.enabled = False
from tools.engine import create_model
import torchvision.transforms as transforms
from tools.utils import compute_miou,compute_iou_per_class
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score

def get_args_parser():
    parser = argparse.ArgumentParser('prediction script', add_help=False)
    parser.add_argument('--batch-size', default=64, type=int)
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
    parser.add_argument('--data-path', type=str, default=r"J:\Train\Train\Rural\img_low",
                        help='data-path')
    parser.add_argument('--label-path', type=str,
                        default=r"J:\Train\Train\Rural\label_low",
                        help='label-path')

    parser.add_argument('--weight-path', type=str,
                        default=r"E:\fcn\fcn\best_UNET_model_pretrained.pth",
                        help='weight-path')
    return parser
rgb_mapping = [
    255, 0, 0,  # 红色 -> 标签 0
    0, 255, 0,  # 绿色 -> 标签 1
    0, 0, 128,  # 蓝色 -> 标签 2
    128, 128, 0,  # 标签 3
    255, 255, 0,  # 标签 4
    0, 128, 128,  # 标签 5
    128, 0, 0,  # 标签 6
    0, 128, 0,  # 标签 7
    # 添加更多映射
]

def main(args,rgb_mapping):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size
    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    # 加载数据
    image_folder = args.data_path
    label_folder = args.label_path
    image_paths = [os.path.join(image_folder, fname) for fname in os.listdir(image_folder)]
    label_paths = [os.path.join(label_folder, fname) for fname in os.listdir(label_folder)]
    image_paths.sort()
    label_paths.sort()
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts image to tensor and scales to [0, 1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize with ImageNet stats
    ])
    dataset = SegmentationDataset(image_paths,  label_paths,transform)
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

    model.load_state_dict(weights_dict)
    model.to(device)

    all_preds = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(data_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            # 进行预测
            output = model(inputs)["out"]
            preds = torch.argmax(output, dim=1)  # 获取预测的类别
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            # 对每个批次中的每个图像进行处理和保存
            for j in range(inputs.size(0)):
                prediction = output[j].argmax(0).to("cpu").numpy().astype(np.uint8)
                mask = Image.fromarray(prediction)
                mask.putpalette(rgb_mapping)

                # 保存预测结果
                original_image_name = os.path.basename(image_paths[ i * batch_size + j]).split('.')[0]
                save_path = os.path.join("output_masks", f"{original_image_name}_mask.png")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)  # 创建目录
                mask.save(save_path)
                print(f"Saved mask {i * batch_size + j} to {save_path}")
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    print(all_labels.shape)
    print(all_labels.shape)
    accuracy = accuracy_score(all_labels, all_preds)
    IoU = compute_iou_per_class(all_preds, all_labels, num_classes=args.num_classes)
    mIoU = compute_miou(all_preds, all_labels, num_classes=args.num_classes)
    m_precision, m_recall, m_F1 = precision_recall_fscore_support(all_labels, all_preds, average='macro')[:-1]
    precision, recall, F1, support = precision_recall_fscore_support(all_labels, all_preds, average=None)
    print(f'OA:{accuracy}')
    print(f'UA:{precision}')
    print(f'mUA:{m_precision}')
    print(f'F1:{F1}')
    print(f'mF1:{m_F1}')
    print(f'IoU:{IoU}')
    print(f'mIoU:{mIoU}')






if __name__ == '__main__':
    parser = argparse.ArgumentParser('prediction script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args,rgb_mapping)
