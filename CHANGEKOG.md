# 24-vision-yhl
The Test Task of The Vision by Yhl

## [1.0.0] -2023-10-01

### Feature 新增

+ 创建了`rectangle`文件夹，添加了装甲板识别的主函数代码和相应的`CMakeLists.txt`文件以及`READMRE.md`文件<font color=#FF0000>（***可作为MarkDown文档撰写的考核作品***）</font>，它计算检测到的物体的平均中心和边界点并在视频帧中标注显示。
+ 创建了`CMake_1-master`文件夹，初步完成了CMake任务一，实现了编译生成可执行文件并成功运行。

## [1.0.1] -2023-10-04

### Feature 新增

+ 添加了装甲板到相机的距离和姿态的检测代码到相应主函数。
+ 优化完善了`CMakeLists.txt`中对OpenCv库的链接。

## [1.0.2] -2023-10-05

### Changed 变更
* 重新配置了`.git`和`.gitignore`文件，使所有变更直接同步到`main`分支，优化了代码上传体验。
* 修改CMake任务一的相关代码，使其在不修改原有任何文件（仅添加）的情况下完成任务。
