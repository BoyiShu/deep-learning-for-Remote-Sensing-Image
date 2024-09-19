import torch
import time
from model import *
import torch.optim as optim
import numpy as np
from osgeo import gdal
gdal.UseExceptions()
from PIL import Image
from tqdm import tqdm
import sys

def create_model(model_name, num_classes,  in_channels=3):
    if model_name=="resnet34":
        model = resnet34(in_channels=in_channels, num_classes=num_classes)
    elif model_name == "resnet50":
        model = resnet50(in_channels=in_channels, num_classes=num_classes)
    elif model_name == "resnet101":
        model = resnet101(in_channels=in_channels, num_classes=num_classes)
    elif model_name == "resnext50_32x4d":
        model = resnext50_32x4d(in_channels=in_channels, num_classes=num_classes)
    elif model_name == "resnext101_32x8d":
        model = resnext101_32x8d(in_channels=in_channels, num_classes=num_classes)

    elif model_name == "AlexNet":
        model = AlexNet(in_channels=in_channels, num_classes=num_classes)
    elif model_name == "vgg11":
        model = vgg11(in_channels=in_channels, num_classes=num_classes)
    elif model_name == "vgg13":
        model = vgg13(in_channels=in_channels, num_classes=num_classes)
    elif model_name == "vgg16":
        model = vgg16(in_channels=in_channels, num_classes=num_classes)
    elif model_name == "vgg19":
        model = vgg19(in_channels=in_channels, num_classes=num_classes)
    elif model_name == "regnet":
        model = create_regnet(in_channels=in_channels, num_classes=num_classes)
    elif model_name == "convnext_tiny":
        model = convnext_tiny(in_channels=in_channels, num_classes=num_classes)
    elif model_name == "convnext_small":
        model = convnext_small(in_channels=in_channels, num_classes=num_classes)
    elif model_name == "convnext_base":
        model = convnext_base(in_channels=in_channels, num_classes=num_classes)
    elif model_name == "convnext_large":
        model = convnext_large(in_channels=in_channels, num_classes=num_classes)
    elif model_name == "convnext_xlarge":
        model = convnext_xlarge(in_channels=in_channels, num_classes=num_classes)
    elif model_name == "efficientnetv2_s":
        model = efficientnetv2_s(in_channels=in_channels, num_classes=num_classes)
    elif model_name == "efficientnetv2_m":
        model = efficientnetv2_m(in_channels=in_channels, num_classes=num_classes)
    elif model_name == "efficientnetv2_l":
        model = efficientnetv2_l(in_channels=in_channels, num_classes=num_classes)
    elif model_name == "shufflenet_v2_x0_5":
        model = shufflenet_v2_x0_5(in_channels=in_channels, num_classes=num_classes)
    elif model_name == "shufflenet_v2_x1_0":
        model = shufflenet_v2_x1_0(in_channels=in_channels, num_classes=num_classes)
    elif model_name == "shufflenet_v2_x1_5":
        model = shufflenet_v2_x1_5(in_channels=in_channels, num_classes=num_classes)
    elif model_name == "shufflenet_v2_x2_0":
        model = shufflenet_v2_x2_0(in_channels=in_channels, num_classes=num_classes)
    elif model_name == "densenet121":
        model = densenet121(in_channels=in_channels, num_classes=num_classes)
    elif model_name == "densenet161":
        model = densenet161(in_channels=in_channels, num_classes=num_classes)
    elif model_name == "densenet169":
        model = densenet169(in_channels=in_channels, num_classes=num_classes)
    elif model_name == "densenet201":
        model = densenet201(in_channels=in_channels, num_classes=num_classes)

    # 其他模型构建逻辑
    return model

def create_optimizer(opt_name,lr_,model):
    if opt_name=="adamw":
      optimizer = optim.Adam(model.parameters(), lr=lr_)
    return optimizer


def train(model, train_loader, test_loader, device, num_epochs, criterion, optimizer, batch_size,weight_path):
    """
    训练模型，并计算每个 epoch 的训练时间
    """
    train_losses = []
    best_accuracy = 0.0
    epoch_times = []  # 创建一个列表来存储每个 epoch 的时间

    for epoch in range(num_epochs):
        start_time = time.time()  # 记录每个 epoch 开始时间

        # 训练过程
        model.train()
        epoch_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.permute(0, 3, 1, 2)
            optimizer.zero_grad()
            output = model(inputs)

            loss = criterion(output, labels.long())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            if (i + 1) % batch_size == 0:  # 每 batch_size 个样本更新一次
                # 计算平均训练损失
                avg_train_loss = epoch_loss / batch_size
                train_losses.append(avg_train_loss)

                # 测试过程
                model.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for inputs, labels in test_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        inputs = inputs.permute(0, 3, 1, 2)
                        outputs = model(inputs)
                        predicted = torch.max(outputs, dim=1)[1]
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()

                accuracy = correct / total

                print(
                    f'Epoch [{epoch + 1}/{num_epochs}] - Batch [{i + 1}/{len(train_loader)}] - Train Loss: {avg_train_loss:.4f} - Test Accuracy: {accuracy * 100:.2f}%')

                # 保存最佳模型
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    torch.save(model.state_dict(), weight_path)

                # 重置 epoch 损失
                epoch_loss = 0.0

        end_time = time.time()  # 记录每个 epoch 结束时间
        epoch_time = end_time - start_time  # 计算每个 epoch 的训练时间
        epoch_times.append(epoch_time)  # 将每个 epoch 的时间添加到列表中
        print(f'Epoch [{epoch + 1}/{num_epochs}] - Time: {epoch_time:.2f}s')

    # 计算所有 epoch 的平均训练时间
    avg_epoch_time = sum(epoch_times) / len(epoch_times)

    print(f"Average Training Time per Epoch: {avg_epoch_time:.2f}s")

    """
       评估模型在测试集上的性能
    """


    return model

def predict(model,full_loader,device,copy_data_path,save_path):
    model.eval()
    predictions = []
    with torch.no_grad():
        for inputs, labels in full_loader:
            # inputs = F.pad(inputs, (0, 0, 6, 7, 6, 7), mode='constant', value=0)
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.permute(0, 3, 1, 2)
            outputs = model(inputs)
            predicted = torch.max(outputs, dim=1)[1]
            predictions.extend(predicted.cpu().numpy())

    # 改变数值类型
    y_pred_img = np.array(predictions).astype(np.uint8)

    # 将预测结果转换为图像形状并且保存
    tag_g_ds = gdal.Open(copy_data_path)
    tag_g_data = tag_g_ds.ReadAsArray()
    tag_g_data_change = tag_g_data.transpose(1, 2, 0)
    height, width, channels = tag_g_data_change.shape
    y_pred_img = y_pred_img.reshape(height, width)

    image = Image.fromarray(y_pred_img)
    image.save(save_path)

    return y_pred_img
def train2(model, train_loader, test_loader, device, num_epochs, criterion, optimizer, batch_size,weight_path,val_num):
    train_losses = []
    best_accuracy = 0.0
    epoch_times = []  # 创建一个列表来存储每个 epoch 的时间
    running_loss=0.0
    train_steps = len(train_loader)
    for epoch in range(num_epochs):
        start_time = time.time()  # 记录每个 epoch 开始时间

        # 训练过程
        model.train()
        epoch_loss = 0.0
        best_acc=0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            logits = model(images.to(device))
            loss = criterion(logits, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     num_epochs,
                                                                     loss)

            # validate
        model.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(test_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = model(val_images.to(device))
                # loss = loss_function(outputs, test_labels)
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                               num_epochs)

        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
                  (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(model.state_dict(), weight_path)
        end_time = time.time()  # 记录每个 epoch 结束时间
        epoch_time = end_time - start_time  # 计算每个 epoch 的训练时间
        epoch_times.append(epoch_time)  # 将每个 epoch 的时间添加到列表中
        print(f'Epoch [{epoch + 1}/{num_epochs}] - Time: {epoch_time:.2f}s')

            # 计算所有 epoch 的平均训练时间
    avg_epoch_time = sum(epoch_times) / len(epoch_times)

    print(f"Average Training Time per Epoch: {avg_epoch_time:.2f}s")

    print('Finished Training')
