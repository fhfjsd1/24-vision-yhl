#!/usr/bin/env python3
#coding:utf-8

import rospy
import cv2
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
# from dynamic_reconfigure.server import Server
# from camera_driver.cfg import CameraConfig

class CameraDriverNode:
    def __init__(self):
        rospy.init_node('camera_driver_node', anonymous=True)
        
        self.image_pub = rospy.Publisher('camera/image', Image, queue_size=10)
        self.camera_info_pub = rospy.Publisher('camera/camera_info', CameraInfo, queue_size=10)
        self.bridge = CvBridge()

        self.camera_params = rospy.get_param('~camera_params', {})

    def config_callback(self, config, level):
        # 动态配置相机参数
        self.camera_params = config
        return config

    def capture_and_publish(self):

        cap = cv2.VideoCapture(0) 

        if not cap.isOpened():
            rospy.logerr("打不开啊啊啊啊啊")
            return

        # 从相机截取视频帧
        ret, frame = cap.read()
        if not ret:
            rospy.logerr("拍不了啊啊啊啊")
            return

        # 转为ROS format
        image_msg = self.bridge.cv2_to_imgmsg(frame, 'bgr8')

        # 装载参数
        camera_info = CameraInfo()
        camera_info.width = frame.shape[1]
        camera_info.height = frame.shape[0]
        # camera_info.K = self.camera_params['K']
        # camera_info.P = self.camera_params['P']
        K_param_str = self.camera_params['K']
        camera_info.K = [float(val) for val in K_param_str.split(',')]
        P_param_str = self.camera_params['P']
        camera_info.P = [float(val) for val in P_param_str.split(',')]


        # 发布信息
        self.image_pub.publish(image_msg)
        self.camera_info_pub.publish(camera_info)

        # 释放
        cap.release()
    def run(self):
        rate = rospy.Rate(10)  # 10Hz
        while not rospy.is_shutdown():
            self.capture_and_publish()
            rate.sleep()

if __name__ == '__main__':
    try:
        driver = CameraDriverNode()
        driver.run()
    except rospy.ROSInterruptException:
        pass
