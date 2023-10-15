# 使用 PyTorch 的 YOLOX 目标检测网络并使用 TensorRT 部署到 C++ 的全流程

## 1. 安装必要的库和工具

首先，确保系统上安装了以下依赖项，库和工具：

- Python 3.8
- PyTorch
- YOLOX
- OpenCV_Python
- TensorRT
- CMake
- g++ 编译器
- torch2trt
- numpy
-torch>=1.7
-loguru
-tqdm
-torchvision
-thop
-ninja
-tabulate
-psutil
-tensorboard

## 2. 下载 YOLOX 预训练模型

从官方GitHub仓库下载基于Pytorch的YOLOX主要源代码和示例代码，此处选用`yolox_s.pth`预训练模型。

## 3. 验证安装

使用自带测试demo，在终端输入命令后输出如下图像：
```
python3 tools/demo.py image -f exps/default/yolox_s.py -c ./yolox_s.pth --path assets/dog.jpg --conf 0.3 --nms 0.65 --tsize 640 --save_result --device gpu
```
![test](https://img-blog.csdnimg.cn/img_convert/56688e5cae3385f3e186e0ba8c8c44e7.png)

## 4. 制作数据集

数据集采用VOC数据集，原始数据集是Labelme标注的数据集。下载地址：https://download.csdn.net/download/hhhhhhhhhhwwwwwwwwww/14003627。
运行脚本后分类整理得到成品数据集。
![datasets](https://img-blog.csdnimg.cn/img_convert/a0b98b86b782c773ec067a1ca2e00613.png)

## 5. 修改配置文件

该部分主要为修改源代码中类别数目及名称，数据集路径等。

## 6. 训练模型

基于预训练模型迁移学习，来训练自己的模型，使用命令行执行操作：
```
python3 tools/train.py -f exps/example/yolox_voc/yolox_voc_s.py -d 1 -b 4 --fp16  -c yolox_s.pth
```

## 7. 测试模型

类似于第五步，修改相关配置文件后使用命令行执行操作：
```
python3 tools/demo.py image -f exps/example/yolox_voc/yolox_voc_s.py -c YOLOX_outputs/yolox_voc_s/latest_ckpt.pth --path ./assets/aircraft_589.jpg --conf 0.3 --nms 0.65 --tsize 640 --save_result --device gpu
```


## 6. 编译 C++ 代码

使用 CMake 和 g++ 编译器来构建 C++ 代码，确保链接正确的 TensorRT 和其他依赖项。

## 7. 运行 C++ 应用程序

运行 C++ 应用程序，它将加载优化的模型并执行目标检测。

这个指南提供了一个高级概述，实际流程可能需要更多的细节和配置，具体步骤可能因环境和需求而有所不同。确保查阅相关的 PyTorch、YOLOX 和 TensorRT 文档，以获取更详细的信息和示例代码。
