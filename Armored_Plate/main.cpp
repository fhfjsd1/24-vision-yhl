#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <vector>
#include <cmath>

using namespace cv;
using namespace std;

int main()
{
    VideoCapture cap("/home/taylor/testcpp/24-vision-yhl-1/rectangle/rec.avi"); // 替换为你的视频文件路径
    if (!cap.isOpened())
    {
        cout << "无法打开视频文件" << endl;
        return -1;
    }

    int codec = cv::VideoWriter::fourcc('X', 'V', 'I', 'D'); // 使用XVID编解码器
    cv::Size frame_size(static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH)),
                        static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT)));
    cv::VideoWriter output("output_video.avi", codec, 30, frame_size); // 输出视频文件名和帧速率
    if (!output.isOpened())
    {
        std::cerr << "无法创建输出视频文件" << std::endl;
        return -1;
    }

    while (true)
    {
        Mat frame;
        cap >> frame; // 读取视频帧

        Mat cameraMatrix = (Mat_<double>(3, 3) << 1000, 0, frame.cols / 2, 0, 1000, frame.rows / 2, 0, 0, 1); // 相机内参矩阵
        Mat distCoeffs = Mat::zeros(5, 1, CV_64F);                                                            // 畸变系数
        Mat rotationVector, translationVector;                                                                // 旋转矩阵和平移矩阵
        double actual_length = 5.2;                                                                           // 实际距离（单位：cm）

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

        // 使用霍夫变换找到黄色线条（好像失败了）
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

        double focalLength = sqrt((cameraMatrix.at<double>(0, 0)) * (cameraMatrix.at<double>(0, 0)) +
                                  (cameraMatrix.at<double>(1, 1)) * (cameraMatrix.at<double>(1, 1))); // 焦距（单位：像素）

        double pixelDistance1 = norm(extremepoints[0] - extremepoints[1]); // 两个像素点之间的像素距离
        double pixelDistance2 = norm(extremepoints[2] - extremepoints[3]); // 两个像素点之间的像素距离
        double depth1 = (actual_length * focalLength) / pixelDistance1;
        double depth2 = (actual_length * focalLength) / pixelDistance1;
        double depth = (depth1 + depth2) / 2;

        // 计算相机坐标系中的三维点
        std::vector<Point3f> objectPoints;
        for (int i = 0; i < 4; ++i)
        {
            objectPoints.push_back(Point3f((extremepoints[i].x - cameraMatrix.at<double>(0, 2)) / cameraMatrix.at<double>(0, 0),
                                           (extremepoints[i].y - cameraMatrix.at<double>(1, 2)) / cameraMatrix.at<double>(1, 1),
                                           depth / 0.04));
        }

        solvePnP(objectPoints, extremepoints, cameraMatrix, distCoeffs, rotationVector, translationVector);

        // // 计算平面的法向量
        // Mat rotationMatrix;
        // Rodrigues(rotationVector, rotationMatrix);
        // Mat normalVector = rotationMatrix.col(2);

        // 计算平面到相机的距离
        // int distance = -normalVector.dot(translationVector);

        const double initial_velocity = 20.0; // 初始速度，单位：米/秒
        const double gravity = 9.81;          // 重力加速度，单位：米/秒^2
        double distance = depth / 10;
        double launch_angle = 0;
        double horizontal_velocity = initial_velocity * cos(launch_angle);
        double vertical_velocity = initial_velocity * sin(launch_angle);
        for (; launch_angle <= 1.57;)
        {
            double t1 = vertical_velocity * 2 / gravity;
            double t2 = distance / horizontal_velocity;
            if (-0.1 <= (t1 - t2) && (t1 - t2) <= 0.1)
                break;
            else
            {
                launch_angle += 0.01;
                horizontal_velocity = initial_velocity * cos(launch_angle);
                vertical_velocity = initial_velocity * sin(launch_angle);
            }
        }

        // 打印结果
        cout << "平面到相机的距离: " << depth << "cm(相关参数为自行设定)" << std::endl;
        cout << "旋转向量: " << rotationVector << std::endl;
        cout << "平移向量: " << translationVector << std::endl;
        cout << "发射仰角：" << launch_angle / 3.14159 * 180 << "度" << endl;

        line(frame, extremepoints[0], extremepoints[3], Scalar(0, 0, 255), 1, 8);
        line(frame, extremepoints[1], extremepoints[2], Scalar(0, 0, 255), 1, 8);

        if (validContours > 0)
        {
            averageCenter /= validContours;                         // 计算平均中心点
            circle(frame, averageCenter, 5, Scalar(0, 0, 255), -1); // 在平均中心点处绘制红色圆点
        }

        std::string text1 = "distance(the relevant parameters are self set):";
        std::string depth_text = to_string(depth);
        std::string unit = "cm";
        std::string distance_text = text1 + depth_text + unit;
        std::string note = "Attitude information (just temporarily) output at the terminal!";
        std::string angletxt = "launch angle(under angle system):";
        std::string angle = to_string(launch_angle / 3.14159 * 180);
        std::string angle_text = angletxt + angle;

        cv::putText(frame, distance_text, cv::Point(20, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
        cv::putText(frame, angle_text, cv::Point(20, 50), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
        cv::putText(frame, note, cv::Point(20, 450), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);

        output << frame; // 将帧写入输出视频

        imshow("Video", frame);

        if (waitKey(30) == 27)
        { // 按下ESC键退出
            cout << "用户按下ESC键，退出程序" << endl;
            break;
        }
    }

    cap.release();
    output.release();
    destroyAllWindows();

    return 0;
}
