from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import torch


class SegmentationDataset(Dataset):
    def __init__(self, image_paths, label_paths, transform=None):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        image = np.array(image, dtype=np.float32)

        # 加载标签图像
        label = Image.open(self.label_paths[idx])
        label = np.array(label, dtype=np.int64)
        # 如果有transform，应用于image和label
        if self.transform:
            image = self.transform(image)

        label = torch.tensor(label)
        # print(f"Label min: {label.min()}, Label max: {label.max()}")


        return image, label


def compute_iou_per_class(preds, labels, num_classes):
    ious = []
    for cls in range(num_classes):
        pred_inds = preds == cls
        label_inds = labels == cls
        if np.sum(label_inds) == 0:
            ious.append(float('nan'))  # Ignore this class if not in the ground truth
        else:
            intersection = np.logical_and(pred_inds, label_inds)
            union = np.logical_or(pred_inds, label_inds)
            iou = np.sum(intersection) / np.sum(union)
            ious.append(iou)
    return ious

def compute_miou(preds, labels, num_classes):
    ious = compute_iou_per_class(preds, labels, num_classes)
    print(ious)
    valid_ious = [iou for iou in ious if not np.isnan(iou)]
    return np.mean(valid_ious)


# preds = np.array([0, 1, 2, 0, 2, 1])
# labels = np.array([0, 1, 1, 0, 2, 2])
# miou = compute_miou(preds, labels, num_classes=3)
# print(miou)