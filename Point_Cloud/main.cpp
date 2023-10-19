#include <iostream>
#include <fstream>
#include <cstring>
#include <vector>
#include <opencv2/opencv.hpp> // 使用OpenCV库

using namespace std;
using namespace cv;

int csv(string location, vector<Point3d> &point_cloud)
{
    string line;
    // 读取CSV文件
    std::ifstream file0(location, ios::in);

    if (!file0.is_open())
    {
        cerr << "无法打开CSV文件" << endl;
        return -1;
    }

    while (getline(file0, line))
    {
        istringstream ss(line);
        double x, y, z;
        char delimiter;
        ss >> x >> delimiter >> y >> delimiter >> z;
        point_cloud.push_back(Point3d(x, y, z));
    }

    file0.close();
    return 0;
}

int main()
{
    std::vector<cv::Point3d> point_cloud;
    string line;
    string root_path="/home/taylor/testcpp/csv/";

    csv(root_path+"one/cloud_0.csv", point_cloud);
    csv(root_path+"one/cloud_1.csv", point_cloud);
    csv(root_path+"one/cloud_2.csv", point_cloud);
    csv(root_path+"one/cloud_3.csv", point_cloud);
    csv(root_path+"one/cloud_4.csv", point_cloud);
    csv(root_path+"two/cloud_0.csv", point_cloud);
    csv(root_path+"two/cloud_1.csv", point_cloud);
    csv(root_path+"two/cloud_2.csv", point_cloud);
    csv(root_path+"two/cloud_3.csv", point_cloud);
    csv(root_path+"two/cloud_4.csv", point_cloud);
    csv(root_path+"three/cloud_0.csv", point_cloud);
    csv(root_path+"three/cloud_1.csv", point_cloud);
    csv(root_path+"three/cloud_2.csv", point_cloud);
    csv(root_path+"three/cloud_3.csv", point_cloud);
    csv(root_path+"three/cloud_4.csv", point_cloud);
    csv(root_path+"forth/cloud_0.csv", point_cloud);
    csv(root_path+"forth/cloud_1.csv", point_cloud);
    csv(root_path+"forth/cloud_2.csv", point_cloud);
    csv(root_path+"forth/cloud_3.csv", point_cloud);
    csv(root_path+"forth/cloud_4.csv", point_cloud);

    // 定义深度图像的大小
    int width = 2000;
    int height = 1300;

    // 创建深度图像
    // cv::Mat outputImage = cv::Mat::zeros(height, width, CV_32F);
    Mat outputImage(height, width, CV_8UC3, cv::Scalar(255,255,255)); // 用白色背景

    // 内部相机参数
    cv::Mat CameraMat = (cv::Mat_<double>(3, 3) << 1.3859739625395162e+03, 0, 9.3622464596653492e+02, 0,
                         1.3815353250336800e+03, 4.9459467170828475e+02, 0, 0, 1);
    cv::Mat dist_coeffs = (cv::Mat_<double>(1, 5) << 7.0444095385902794e-02, -1.8010798300183417e-01,
                           -7.7001990711544465e-03, -2.2524968464184810e-03,
                           1.4838608095798808e-01);
    Mat cameraExtrinsicMat = (Mat_<double>(4, 4) << -7.1907391850483116e-03, 1.0494953004635377e-02,
                              9.9991907134097757e-01, 1.0984281510814174e-01,
                              -9.9997142335005851e-01, 2.2580773589691017e-03,
                              -7.2148159989590677e-03, -1.8261670813403203e-02,
                              -2.3336137706425064e-03, -9.9994237686382270e-01,
                              1.0478415848689249e-02, 1.7323651488230618e-01, 0., 0., 0., 1.);

    Mat rotationMatrix = cameraExtrinsicMat(Range(0, 3), Range(0, 3));
    Mat translationVector = cameraExtrinsicMat.col(3).rowRange(0, 3);
    Mat inversemat;
    Mat negatevec = (-1.) * translationVector;
    invert(rotationMatrix, inversemat);
    Mat rotationVector;
    Rodrigues(inversemat, rotationVector);

    // 转换三维点到相机坐标系
    vector<cv::Point2d> point_2d;
    cv::projectPoints(point_cloud, rotationVector, negatevec, CameraMat, dist_coeffs, point_2d);

    for (int i = 0; i < point_2d.size(); i++)
    {
        cout << "point_cloud" << point_cloud[i] << "  "
             << "point_2d" << point_2d[i] << endl;
        circle(outputImage, point_2d[i], 3, Scalar(0,0,0), -1);
    }
    cv::imwrite("outputImage.png", outputImage);

    return 0;
}
