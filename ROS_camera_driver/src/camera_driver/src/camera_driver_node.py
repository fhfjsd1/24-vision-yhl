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

        #self.server = Server(CameraConfig, self.config_callback)
        self.camera_params = rospy.get_param('~camera_params', {})

    def config_callback(self, config, level):
        # Update camera parameters when dynamic reconfigure is called
        self.camera_params = config
        return config

    def capture_and_publish(self):
        # Capture image from the built-in camera
        cap = cv2.VideoCapture(0)  # 0 represents the default camera (you may need to adjust this)

        if not cap.isOpened():
            rospy.logerr("Failed to open the camera.")
            return

        # Capture a frame from the camera
        ret, frame = cap.read()
        if not ret:
            rospy.logerr("Failed to capture an image from the camera.")
            return

        # Convert the captured frame to ROS format
        image_msg = self.bridge.cv2_to_imgmsg(frame, 'bgr8')

        # Populate CameraInfo message with parameters
        camera_info = CameraInfo()
        camera_info.width = frame.shape[1]
        camera_info.height = frame.shape[0]
        # camera_info.K = self.camera_params['K']
        # camera_info.P = self.camera_params['P']
        K_param_str = self.camera_params['K']
        camera_info.K = [float(val) for val in K_param_str.split(',')]
        P_param_str = self.camera_params['P']
        camera_info.P = [float(val) for val in P_param_str.split(',')]


        # Publish the image and camera info
        self.image_pub.publish(image_msg)
        self.camera_info_pub.publish(camera_info)

        # Release the camera
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
