import time
import torch
from model import fcn_resnet50,VGG16UNet
from model.deeplabv3_model import deeplabv3_resnet50
import torch.optim as optim
from torch import nn
from torch.nn.functional import cross_entropy as criterion
import numpy as np
from utils import compute_miou,compute_iou_per_class
import datetime

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score

def criterion_(inputs, target):
    losses = {}
    for name, x in inputs.items():
        # 忽略target中值为255的像素，255的像素是目标边缘或者padding填充
        losses[name] = nn.functional.cross_entropy(x, target, ignore_index=255)

    if len(losses) == 1:
        return losses['out']

    return losses['out'] + 0.5 * losses['aux']

def create_model(aux,model_name, num_classes,):
    if model_name=="fcn_resnet50":
      model = fcn_resnet50(aux,num_classes=num_classes)
    elif model_name == "unet":
      model = UNet(num_classes=num_classes,pretrain_backbone=True)
    elif model_name == "vgg16unet":
      model = VGG16UNet(num_classes=num_classes,pretrain_backbone=True)
    elif model_name == "deeplabv3_resnet50":
      model = deeplabv3_resnet50(aux,num_classes=num_classes,pretrain_backbone=True)
    # 其他模型构建逻辑
    return model

def create_optimizer(opt_name,lr_,model):
    if opt_name=="adamw":
      optimizer = optim.Adam(model.parameters(), lr=lr_)
    return optimizer


def train(num_epochs,dataloader,val_dataloader,optimizer,model,aux,weight_path,device,batch_size,num_classes,lr_scheduler):
    epoch_times = []
    train_losses = []
    best_accuracy = 0.0
    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    model.train()
    for epoch in range(num_epochs):
        start_time = time.time()
        running_loss = 0.0
        for i, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()  # 清零梯度
            outputs = model(images)  # 前向传播

            if aux:
                loss = criterion_(outputs, labels)
            else:
                loss = criterion(outputs["out"], labels) # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新权重
            lr_scheduler.step()
            running_loss += loss.item()

            if (i + 1) % batch_size == 0 or (i + 1) == len(dataloader):
                avg_train_loss = running_loss / (i+1)
                train_losses.append(avg_train_loss)

                model.eval()  # 设置为评估模式
                all_preds = []
                all_labels = []

                with torch.no_grad():
                    for images, labels in val_dataloader:
                        images, labels = images.to(device), labels.to(device)
                        outputs = model(images)['out']


                        preds = torch.argmax(outputs, dim=1)  # 获取预测的类别
                        all_preds.append(preds.cpu().numpy())
                        all_labels.append(labels.cpu().numpy())




                all_preds = np.concatenate(all_preds)
                all_labels = np.concatenate(all_labels)
                accuracy = accuracy_score(all_labels, all_preds)
                m_precision, m_recall, m_F1 = precision_recall_fscore_support(all_labels, all_preds, average='macro')[:-1]
                precision, recall, F1 ,support= precision_recall_fscore_support(all_labels, all_preds, average=None)

                IoU = compute_iou_per_class(all_preds, all_labels, num_classes=num_classes)
                mIoU = compute_miou(all_preds, all_labels, num_classes=num_classes)

                ua_str = ', '.join([f'{ua:.4f}' for ua in precision])
                f1_str = ', '.join([f'{f1:.4f}' for f1 in F1])
                iou_str = ', '.join([f'{iou:.4f}' for iou in IoU])

                with open(results_file, "a") as f:
                    train_info = f"[epoch: {epoch+1}]\n" \
                             f"Batch: {i + 1}/{len(dataloader)}\n" \
                             f"Train Loss: {avg_train_loss:.4f}\n" \
                             f"OA: {accuracy:.4f}\n" \
                             f"UA: {ua_str}\n" \
                             f"mUA: {m_precision:.4f}\n" \
                             f"F1: {f1_str}\n" \
                             f"mF1: {m_F1:.4f}\n" \
                             f"IoU:{iou_str}\n"  \
                             f"mIoU: {mIoU:.4f}\n"
                    f.write(train_info + "\n\n")
                print(
                f'Epoch [{epoch + 1}/{num_epochs}] -Batch [{i + 1}/{len(dataloader)}] -Train Loss: {avg_train_loss:.4f} -OA: {accuracy:.4f} -mUA: {m_precision:.4f} -mF1: {m_F1:.4f} -mIoU: {mIoU:.4f} -IoU:{iou_str}')
                if best_accuracy < mIoU:
                    best_accuracy = mIoU
                    torch.save(model.state_dict(), weight_path)
        end_time = time.time()  # 记录每个 epoch 结束时间
        epoch_time = end_time - start_time  # 计算每个 epoch 的训练时间
        epoch_times.append(epoch_time)  # 将每个 epoch 的时间添加到列表中
        print(f'Epoch [{epoch + 1}/{num_epochs}] - Time: {epoch_time:.2f}s')
        with open(results_file, "a") as f:
            epoch_one_time=f"[epoch: {epoch+1}]\n" \
                           f"Time: {epoch_time:.2f}s\n"
            f.write(epoch_one_time + "\n\n")
                # 计算所有 epoch 的平均训练时间
    avg_epoch_time = sum(epoch_times) / len(epoch_times)

    print(f"Average Training Time per Epoch: {avg_epoch_time:.2f}s")
    with open(results_file, "a") as f:
        epoch_one_time =f"{avg_epoch_time:.2f}\n"
        f.write(epoch_one_time + "\n\n")


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            # warmup后lr倍率因子从1 -> 0
            # 参考deeplab_v2: Learning rate policy
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)