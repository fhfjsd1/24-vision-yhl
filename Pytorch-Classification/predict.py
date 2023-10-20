import torch
# from train_resnet import SelfNet
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
    # train_transforms = data_transforms['train']
    valid_transforms = data_transforms['val']
    # 加载网络
    model = SELFMODEL(model_name=model_name, out_features=num_classes, pretrained=False)
    # model = nn.DataParallel(model)
    weights = torch.load(model_path)
    model.load_state_dict(weights)
    model.eval()
    model.to(device)

    # 读取图片 
    img = Image.open(image_path)
    img = valid_transforms(img)
    img = img.unsqueeze(0)
    img = img.to(device)
    output = model(img)
    label_id = torch.argmax(output).item()
    predict_name = classes_names[label_id]
    print(f"{image_path}'s result is {predict_name}")


if __name__ == '__main__':
    # 批量预测函数
    # predict_batch(model_path=model_path,
    #               target_dir="目标路径",
    #               save_dir="结果路径")

    # 单张图片预测函数
    predict_single(model_path=model_path, image_path="/home/taylor/testcpp/24-vision-yhl-1/Pytorch-Classification/data/validation/2/1502.png")# todo
    predict_single(model_path=model_path, image_path="/home/taylor/testcpp/24-vision-yhl-1/Pytorch-Classification/data/validation/5/1501.png")# todo
    predict_single(model_path=model_path, image_path="/home/taylor/testcpp/24-vision-yhl-1/Pytorch-Classification/data/validation/3/1503.png")# todo