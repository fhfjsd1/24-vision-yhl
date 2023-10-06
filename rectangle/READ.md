# 使用OpenCV进行视频分析

这是一个简单的C++程序，使用OpenCV库执行视频分析。它读取一个视频文件，检测每一帧中的黄色物体，并跟踪它们的位置。同时，它计算检测到的物体的平均中心和边界点，以及其相对相机的距离和姿态，并在视频帧和终端中显示。

## 先决条件

在运行代码之前，您需要安装以下先决条件：

- [OpenCV](https://opencv.org/)（开源计算机视觉库） - 您可以使用软件包管理器安装它，或者按照OpenCV网站上的安装说明进行安装。

## 用法

1. 克隆此存储库或下载代码文件。

2. 使用C++编译器编译代码并与OpenCV链接。例如：

   ```
   g++ video_analysis.cpp -o video_analysis `pkg-config --cflags --libs opencv4`
   ```

3. 运行编译后的二进制文件：

   ```
   ./video_analysis
   ```

4. 程序将打开一个视频文件(***用您自己的视频文件路径替换`/home/taylor/testcpp/rectangle/rec.avi`***），并逐帧显示视频。视频中的黄色物体将被检测并用红色点标记。这些物体的平均中心也将显示为红色点。

5. 按下"ESC"键退出程序。

## 配置

您可以在代码中配置一些参数以满足您的需求：

- `VideoCapture cap("/home/taylor/testcpp/rectangle/rec.avi");`：将路径替换为您的视频文件路径。

- 您可以通过更改`lower_yellow`和`upper_yellow`的值来调整检测黄色物体的HSV颜色范围：

   ```
   Scalar lower_yellow(20, 100, 100);
   Scalar upper_yellow(30, 255, 255);
   ```

- 您可以修改最小轮廓面积阈值，以考虑更大或更小的物体：

   ```
   if (area > 100) { // 根据需要调整阈值
   ```

## 依赖项

- OpenCV

## 许可证

本项目根据MIT许可证授权 - 有关详细信息，请参阅[LICENSE](LICENSE)文件。

## 致谢

此代码是使用OpenCV进行视频分析的简单示例，仅供教育学习目的。它可以进一步扩展和定制以适用于特定应用。本自述文档基于模板编写。
