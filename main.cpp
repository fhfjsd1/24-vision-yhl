#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

int main()
{
    VideoCapture cap("/home/taylor/testcpp/rectangle/rec.avi"); // 替换为你的视频文件路径
    if (!cap.isOpened())
    {
        cout << "无法打开视频文件" << endl;
        return -1;
    }

    while (true)
    {
        Mat frame;
        cap >> frame; // 读取视频帧
        
        Mat cameraMatrix = (Mat_<double>(3, 3) << 1000, 0, frame.cols / 2, 0, 1000, frame.rows / 2, 0, 0, 1); // 相机内参矩阵
        Mat distCoeffs = Mat::zeros(5, 1, CV_64F); // 畸变系数

        if (frame.empty())
        {
            cout << "视频文件结束" << endl;
            break;
        }

        Mat hsv;
        cvtColor(frame, hsv, COLOR_BGR2HSV); // 转换为HSV颜色空间

        // 定义黄色的HSV范围
        Scalar lower_yellow(20, 100, 100);
        Scalar upper_yellow(30, 255, 255);

        Mat yellow_mask;
        inRange(hsv, lower_yellow, upper_yellow, yellow_mask); // 创建黄色掩码

        // 执行形态学操作以去除噪声
        Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
        morphologyEx(yellow_mask, yellow_mask, MORPH_CLOSE, kernel);

        // 找到黄色线条
        // vector<Vec4i> lines;

        // HoughLinesP(yellow_mask, lines, 1, CV_PI / 180, 100, 50, 0);

        // for (size_t i = 0; i < lines.size(); i++) {
        //     Vec4i line = lines[i];
        //     Point startPoint(line[0], line[1]);
        //     Point endPoint(line[2], line[3]);

        //     // 绘制线条
        //      Scalar lineColor(0, 0, 255);
        //      line(frame,startPoint,endPoint,lineColor,2); // 绘制红色线条
        //   // extremepoints.push_back(startPoint);
        //     // extremepoints.push_back(endPoint);

        //     // 绘制端点
        //     circle(frame, startPoint, 5, Scalar(0, 255, 0), -1); // 绘制绿色圆点
        //     circle(frame, endPoint, 5, Scalar(0, 255, 0), -1); // 绘制绿色圆点
        // }

        // 查找轮廓
        vector<vector<Point>> contours;
        vector<Point2f> extremepoints;

        findContours(yellow_mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        Point2f averageCenter(0, 0); // 用于存储平均中心点的坐标
        int validContours = 0;       // 记录有效轮廓的数量

        for (size_t i = 0; i < contours.size(); i++)
        {
            double area = contourArea(contours[i]);
            if (area > 100)
            { // 仅考虑较大的轮廓
                Moments mu = moments(contours[i]);
                Point2f center(mu.m10 / mu.m00, mu.m01 / mu.m00); // 计算轮廓中心
                averageCenter += center;
                validContours++;
            }

            if ((!contours[i].empty()) && (area > 100))
            {
                Point maxYPoint = contours[i][0]; // 初始化最大y坐标的点为第一个点

                for (const Point &p : contours[i])
                {
                    if (p.y > maxYPoint.y)
                    {
                        maxYPoint = p;
                    }
                }
                circle(frame, maxYPoint, 5, Scalar(0, 0, 255), -1);
                extremepoints.push_back(maxYPoint);
            }
            if ((!contours[i].empty()) && (area > 100))
            {
                Point minYPoint = contours[i][0]; // 初始化最大y坐标的点为第一个点

                for (const Point &p1 : contours[i])
                {
                    if (p1.y < minYPoint.y)
                    {
                        minYPoint = p1;
                    }
                }
                circle(frame, minYPoint, 5, Scalar(0, 0, 255), -1);
                extremepoints.push_back(minYPoint);
            }

            /* if (area > 100)
            { // 仅考虑较大的轮廓
                 RotatedRect rect = minAreaRect(contours[i]);
                 Point2f corners[4];
                 rect.points(corners);
                 vector<Point2f> _extremepoints;
                 _extremepoints.push_back((corners[0]+corners[1])/2);
                 _extremepoints.push_back((corners[2]+corners[3])/2);

                 for (int j = 0; j < 2; j++)
                 {
                     extremepoints.push_back(_extremepoints[j]);
                 }
             }*/
        }

        // Mat rvec, tvec; // 旋转向量和平移向量
        // cv::solvePnP(extremepoints, extremepoints, cameraMatrix, distCoeffs, rvec, tvec);

        // // 打印结果
        // std::cout << "平移向量 (tvec): " << tvec << std::endl;
        // std::cout << "旋转向量 (rvec): " << rvec << std::endl;

        line(frame, extremepoints[0], extremepoints[3], Scalar(0, 0, 255), 1, 8);
        line(frame, extremepoints[1], extremepoints[2], Scalar(0, 0, 255), 1, 8);

        if (validContours > 0)
        {
            averageCenter /= validContours;                         // 计算平均中心点
            circle(frame, averageCenter, 5, Scalar(0, 0, 255), -1); // 在平均中心点处绘制红色圆点
        }

        /*  line(frame, extremepoints[0],extremepoints[3] , Scalar(0, 0, 255), 2); // 绘制红色线条
            line(frame, extremepoints[1],extremepoints[2] , Scalar(0, 0, 255), 2); // 绘制红色线条

       // 在矩形上标注端点
         for (size_t i = 0; i < extremepoints.size(); i++)
         {
             circle(frame, extremepoints[i], 5, Scalar(0, 255, 0), -1); // 绘制绿色圆点
         }*/

        imshow("Video", frame);

        if (waitKey(30) == 27)
        { // 按下ESC键退出
            cout << "用户按下ESC键，退出程序" << endl;
            break;
        }
    }

    cap.release();
    destroyAllWindows();

    return 0;
}
