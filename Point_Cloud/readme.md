# 3D 点云到图像坐标的转换

这是一个简单的C++程序，用于将3D点云数据投影到2D图像坐标中。它依赖于OpenCV库，用于读取CSV文件、进行矩阵操作以及图像生成。

## 代码思路

1. **CSV文件读取**: 通过`csv`函数，首先读取一系列CSV文件中的3D点云数据，并将其存储在`std::vector`中。

2. **图像初始化**: 初始化一个指定大小的图像，作为输出图像。在本例中，使用了2000x1300像素大小的白色背景图像。

3. **相机参数定义**: 定义了相机的内部参数、畸变参数和外部参数（旋转矩阵和平移矢量）。

4. **3D到2D转换**: 利用相机参数和3D点云数据，使用`cv::projectPoints`函数将世界坐标系的3D点云投影到像素坐标系的2D图像中。

5. **绘制点**: 遍历投影后的2D点，并在输出图像上绘制这些点。

6. **保存输出图像**: 最后，使用OpenCV的`cv::imwrite`函数将生成的图像保存为PNG文件。

## 编写过程遇到的问题和解决方法

### 1. CSV文件读取

**问题**: 如何将CSV格式文件存储的点云数据提取出来存储成三维点数组。

**解决方法**: 有很多关于CSV数据处理的开源库，可以高效地处理，但鉴于此处数据量较小，遂自行编写简单程序：在`csv`函数中，使用`getline(file0, line)`获取文件的每一行内容，再使用`istringstream`类接收，依据逗号分割数据。

### 2. 相机参数和3D点云数据

**问题**: 相机参数和3D点云数据必须准确匹配，否则会导致投影错误。

**解决方法**: 确保正确设置相机内部参数、畸变参数和外部参数，并确保3D点云数据是与相机坐标系匹配的。如果不匹配，需要进行坐标变换或调整参数。
理论上从外参矩阵可获取世界坐标系与相机坐标系相互变换的旋转矩阵（或旋转向量）和平移向量，通过OpenCV4提供的`cv::projectPoints`函数可以将世界坐标系的3D点云投影到像素坐标系，
返回值为像素坐标系下的各点的像素坐标。尝试对旋转矩阵求逆矩阵或平移向量求反向量，可能修正匹配问题。下面提供官方文档对`cv::projectPoints`函数的完整解释：
```
/** @brief Projects 3D points to an image plane.

@param objectPoints Array of object points, 3xN/Nx3 1-channel or 1xN/Nx1 3-channel (or
vector\<Point3f\> ), where N is the number of points in the view.
@param rvec Rotation vector. See Rodrigues for details.
@param tvec Translation vector.
@param cameraMatrix Camera matrix \f$A = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{_1}\f$ .
@param distCoeffs Input vector of distortion coefficients
\f$(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6 [, s_1, s_2, s_3, s_4[, \tau_x, \tau_y]]]])\f$ of
4, 5, 8, 12 or 14 elements. If the vector is empty, the zero distortion coefficients are assumed.
@param imagePoints Output array of image points, 1xN/Nx1 2-channel, or
vector\<Point2f\> .
@param jacobian Optional output 2Nx(10+\<numDistCoeffs\>) jacobian matrix of derivatives of image
points with respect to components of the rotation vector, translation vector, focal lengths,
coordinates of the principal point and the distortion coefficients. In the old interface different
components of the jacobian are returned via different output parameters.
@param aspectRatio Optional "fixed aspect ratio" parameter. If the parameter is not 0, the
function assumes that the aspect ratio (*fx/fy*) is fixed and correspondingly adjusts the jacobian
matrix.

The function computes projections of 3D points to the image plane given intrinsic and extrinsic
camera parameters. Optionally, the function computes Jacobians - matrices of partial derivatives of
image points coordinates (as functions of all the input parameters) with respect to the particular
parameters, intrinsic and/or extrinsic. The Jacobians are used during the global optimization in
calibrateCamera, solvePnP, and stereoCalibrate . The function itself can also be used to compute a
re-projection error given the current intrinsic and extrinsic parameters.

@note By setting rvec=tvec=(0,0,0) or by setting cameraMatrix to a 3x3 identity matrix, or by
passing zero distortion coefficients, you can get various useful partial cases of the function. This
means that you can compute the distorted coordinates for a sparse set of points or apply a
perspective transformation (and also compute the derivatives) in the ideal zero-distortion setup.
 */
```
