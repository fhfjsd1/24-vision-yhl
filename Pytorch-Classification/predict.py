import torch
from train import SELFMODEL
import os
import os.path as osp
import shutil
import torch.nn as nn
from PIL import Image
from torchutils import get_torch_transforms

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

model_path = "/home/taylor/testcpp/24-vision-yhl-1/Pytorch-Classification/model/resnet50d_10epochs_accuracy1.00000_weights.pth"  # todo  模型路径
classes_names = ['0', '1', '2', '3', '4','5']  # todo 类名
img_size = 224  # todo 图片大小
model_name = "resnet50d"  # todo 模型名称
num_classes = len(classes_names)  # todo 类别数目


def predict_batch(model_path, target_dir, save_dir):
    data_transforms = get_torch_transforms(img_size=img_size)
    valid_transforms = data_transforms['val']
    # 加载网络
    model = SELFMODEL(model_name=model_name, out_features=num_classes, pretrained=False)
    # model = nn.DataParallel(model)
    weights = torch.load(model_path)
    model.load_state_dict(weights)
    model.eval()
    model.to(device)
    # 读取图片
    image_names = os.listdir(target_dir)
    for i, image_name in enumerate(image_names):
        image_path = osp.join(target_dir, image_name)
        img = Image.open(image_path)
        img = valid_transforms(img)
        img = img.unsqueeze(0)
        img = img.to(device)
        output = model(img)
        label_id = torch.argmax(output).item()
        predict_name = classes_names[label_id]
        save_path = osp.join(save_dir, predict_name)
        if not osp.isdir(save_path):
            os.makedirs(save_path)
        shutil.copy(image_path, save_path)
        print(f"{i + 1}: {image_name} result {predict_name}")


def predict_single(model_path, image_path):
    data_transforms = get_torch_transforms(img_size=img_size)
    valid_transforms = data_transforms['val']
    # 加载网络
    model = SELFMODEL(model_name=model_name, out_features=num_classes, pretrained=False)
    weights = torch.load(model_path)
    model.load_state_dict(weights) # 将 weights 中的参数加载到 model 中
    model.eval() # 将模型设置为评估模式
    model.to(device)
    # 读取图片 
    img = Image.open(image_path)
    img = valid_transforms(img)
    img = img.unsqueeze(0) # PyTorch 操作，用于在张量的第一个维度（通常是 batch 维度）上添加一个新的维度，
                           #将其从形状 (H, W, C) 调整为 (1, H, W, C)。这是因为深度学习模型通常要求输入数据以批次的形式，
                           # 即多个样本一起进行处理。这里将单个图像包装在一个批次中，以满足模型的输入要求。
    img = img.to(device)  # 将图像张量移动到设备上
    output = model(img)
    label_id = torch.argmax(output).item() # 在输出张量中找到最高值的索引
    predict_name = classes_names[label_id]
    print(f"{image_path}大概率我觉得应该是{predict_name}")


if __name__ == '__main__':
    # 批量预测函数
    # predict_batch(model_path=model_path,
    #               target_dir="目标路径",
    #               save_dir="结果路径")

    # 单张图片预测函数
    predict_single(model_path=model_path, image_path="/home/taylor/testcpp/24-vision-yhl-1/Pytorch-Classification/data/validation/2/1502.png")# todo
    predict_single(model_path=model_path, image_path="/home/taylor/testcpp/24-vision-yhl-1/Pytorch-Classification/data/validation/5/1501.png")# todo
    predict_single(model_path=model_path, image_path="/home/taylor/testcpp/24-vision-yhl-1/Pytorch-Classification/data/validation/3/1503.png")# todo