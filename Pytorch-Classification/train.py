#准备所需的库和模块
from torchutils import *
from torchvision import datasets, models, transforms
import os.path as osp
import os #os 是Python标准库中的一个模块，用于与操作系统进行交互，它提供了丰富的功能来执行文件、目录、环境变量、进程等操作。

#选择适当的计算设备（GPU或CPU），并将设备赋值给device变量。如果GPU可用，将使用GPU，否则将使用CPU。
if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')
print(f'Using device: {device}')

# 固定随机种子，保证实验结果是可以复现的
seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)  # 设置Python的哈希种子
np.random.seed(seed)  # 设置NumPy的随机数种子
torch.manual_seed(seed)  # 设置PyTorch的随机数种子
torch.cuda.manual_seed(seed)  # 设置PyTorch的CUDA随机数种子
torch.backends.cudnn.deterministic = True  # 设置CuDNN以确保结果的确定性
torch.backends.cudnn.benchmark = True  # 启用CuDNN的性能优化
data_path = "/home/taylor/testcpp/24-vision-yhl-1/Pytorch-Classification/data" # todo 数据集路径

# 超参数设置
params = {
    'model': 'resnet50d',  # 选择预训练模型
    "img_size": 224,  # 图片输入大小
    "train_dir": osp.join(data_path, "train"),  # todo 训练集路径
    "val_dir": osp.join(data_path, "validation"),  # todo 验证集路径
    'device': device,  # 设备
    'lr': 1e-3,  # 学习率
    'batch_size': 4,  # 批次大小
    'num_workers': 0,  # 进程
    'epochs': 10,  # 轮数
    "save_dir": "./checkpoints/",  # todo 保存路径
    "pretrained": False,
    "num_classes": len(os.listdir(osp.join(data_path, "train"))),  # 类别数目, 自适应获取类别数目
    'weight_decay': 1e-5  # 学习率衰减
}

# 定义模型
class SELFMODEL(nn.Module):
    def __init__(self, model_name=params['model'], out_features=params['num_classes'],
                 pretrained=True):
        super().__init__()# 调用父类构造函数
        # 在下面可以初始化子类的属性
        # 使用指定的模型名称从预训练库中创建模型
        self.model = timm.create_model(model_name, pretrained=pretrained, checkpoint_path="/home/taylor/testcpp/24-vision-yhl-1/Pytorch-Classification/pretrained/resnet50d_ra2-464e36ba.pth")
        n_features = self.model.fc.in_features  # 获取要修改的最后那个全连接层输入特征数目
        self.model.fc = nn.Linear(n_features, out_features)  # 修改为本任务对应的类别数目
        # resnet修改最后的全链接层
        print(self.model)  # 返回模型

    def forward(self, x):  # 执行前向传播，将输入数据通过模型传递
        x = self.model(x)
        return x

# 定义训练流程函数
def train(train_loader, model, criterion, optimizer, epoch, params):
    metric_monitor = MetricMonitor()  # 设置指标监视器

    model.train()  # 将模型设置为训练模式

    nBatch = len(train_loader)  # 获取训练数据集的批次数
    stream = tqdm(train_loader)  # 创建进度条对象以可视化训练进度

    for i, (images, target) in enumerate(stream, start=1):  # 开始训练循环
        images = images.to(params['device'], non_blocking=True)  # 将图像数据加载到指定的计算设备
        target = target.to(params['device'], non_blocking=True)  # 加载模型
        output = model(images)  # 将数据传递给模型进行前向传播
        loss = criterion(output, target.long())  # 计算损失
        f1_macro = calculate_f1_macro(output, target)  # 计算F1分数（宏平均）
        recall_macro = calculate_recall_macro(output, target)  # 计算召回率分数（宏平均）
        acc = accuracy(output, target)  # 计算准确率分数
        
        # 进度条更新
        metric_monitor.update('Loss', loss.item())  # 更新损失指标
        metric_monitor.update('F1', f1_macro)  # 更新F1指标
        metric_monitor.update('Recall', recall_macro)  # 更新召回率指标
        metric_monitor.update('Accuracy', acc)  # 更新准确率指标

        optimizer.zero_grad()  # 清空优化器的梯度信息
        loss.backward()  # 反向传播损失
        optimizer.step()  # 更新优化器的权重
        lr = adjust_learning_rate(optimizer, epoch, params, i, nBatch)  # 根据指定策略调整学习率
        stream.set_description(  # 更新进度条显示信息
            "Epoch: {epoch}. Train.      {metric_monitor}".format(
                epoch=epoch,
                metric_monitor=metric_monitor)
        )

    return metric_monitor.metrics['Accuracy']["avg"], metric_monitor.metrics['Loss']["avg"]  # 返回训练的准确率和损失


# 定义验证流程
def validate(val_loader, model, criterion, epoch, params):
    metric_monitor = MetricMonitor()  # 验证流程
    model.eval()  # 模型设置为验证格式
    stream = tqdm(val_loader)  # 设置进度条
    with torch.no_grad():  # 开始推理
        for i, (images, target) in enumerate(stream, start=1):
            images = images.to(params['device'], non_blocking=True)  # 读取图片
            target = target.to(params['device'], non_blocking=True)  # 读取标签
            output = model(images)  # 前向传播
            loss = criterion(output, target.long())  # 计算损失
            f1_macro = calculate_f1_macro(output, target)  # 计算f1分数
            recall_macro = calculate_recall_macro(output, target)  # 计算recall分数
            acc = accuracy(output, target)  # 计算acc
            metric_monitor.update('Loss', loss.item())  # 后面基本都是更新进度条的操作
            metric_monitor.update('F1', f1_macro)
            metric_monitor.update("Recall", recall_macro)
            metric_monitor.update('Accuracy', acc)
            stream.set_description(
                "Epoch: {epoch}. Validation. {metric_monitor}".format(
                    epoch=epoch,
                    metric_monitor=metric_monitor)
            )
    # 返回验证的准确率和损失
    return metric_monitor.metrics['Accuracy']["avg"], metric_monitor.metrics['Loss']["avg"]


# 展示训练过程的曲线
def show_loss_acc(acc, loss, val_acc, val_loss, sava_dir):
    # 从history中提取模型训练集和验证集准确率信息和误差信息
    # 按照上下结构将图画输出
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()), 1])
    plt.title('Training and Validation Accuracy')
    #啊啊啊啊大概率没人看吧啊啊啊啊小发一疯有益身心健康
    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    # 保存在savedir目录下。
    save_path = osp.join(save_dir, "results.png")
    plt.savefig(save_path, dpi=100)


if __name__ == '__main__':
    accs = []
    losss = []
    val_accs = []
    val_losss = []
    data_transforms = get_torch_transforms(img_size=params["img_size"])  # 获取图像预处理方式
    train_transforms = data_transforms['train']  # 训练集数据处理方式
    valid_transforms = data_transforms['val']  # 验证集数据集处理方式
    train_dataset = datasets.ImageFolder(params["train_dir"], train_transforms)  # 加载训练集
    valid_dataset = datasets.ImageFolder(params["val_dir"], valid_transforms)  # 加载验证集

    if params['pretrained'] == True:
        save_dir = osp.join(params['save_dir'], params['model']+"_pretrained_" + str(params["img_size"]))  # 设置模型保存路径
    else:
        save_dir = osp.join(params['save_dir'], params['model'] + "_nopretrained_" + str(params["img_size"]))  # 设置模型保存路径
    if not osp.isdir(save_dir):  # 如果保存路径不存在的话就创建
        os.makedirs(save_dir)  
        print("save dir {} created".format(save_dir))

    train_loader = DataLoader(  # 按照批次加载训练集，
        #batch_size 决定了每个训练批次的大小，shuffle 决定了数据是否洗牌，num_workers 和 pin_memory 用于提高数据加载性能。
        train_dataset, batch_size=params['batch_size'], shuffle=True,
        num_workers=params['num_workers'], pin_memory=True,
    )
    val_loader = DataLoader(  # 按照批次加载验证集
        valid_dataset, batch_size=params['batch_size'], shuffle=False,
        num_workers=params['num_workers'], pin_memory=True,
    )
    print(train_dataset.classes)
    model = SELFMODEL(model_name=params['model'], out_features=params['num_classes'],
                 pretrained=params['pretrained']) # 加载模型
    model = nn.DataParallel(model)  # 模型并行化，提高模型的速度
    model = model.to(params['device'])  # 模型部署到设备上

    criterion = nn.CrossEntropyLoss().to(params['device'])  # 设置损失函数
    # 设置优化器，优化器根据损失函数的梯度来更新模型参数，以减小损失函数的值
    optimizer = torch.optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay']) 
    
    best_acc = 0.0  # 记录最好的准确率
    # 只保存最好的那个模型。
    for epoch in range(1, params['epochs'] + 1):  # 开始训练
        acc, loss = train(train_loader, model, criterion, optimizer, epoch, params)
        val_acc, val_loss = validate(val_loader, model, criterion, epoch, params)
        # 将训练和验证参数存到表中
        accs.append(acc)
        losss.append(loss)
        val_accs.append(val_acc)
        val_losss.append(val_loss)
        if val_acc >= best_acc:
            # 按照目前的情况，如果前面的模型比后面的效果好，就保存一下。
            save_path = osp.join(save_dir, f"{params['model']}_{epoch}epochs_accuracy{acc:.5f}_weights.pth")
            torch.save(model.state_dict(), save_path)
            best_acc = val_acc
    show_loss_acc(accs, losss, val_accs, val_losss, save_dir)
    print("训练已完成，模型和训练日志保存在: {}".format(save_dir))
