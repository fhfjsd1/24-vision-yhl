#include "ros/ros.h"
#include "std_msgs/String.h"

void callbackFunction(const std_msgs::String::ConstPtr& msg) {
  ROS_INFO("I heard: %s", msg->data.c_str());
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "sub_node_name");
  ros::NodeHandle nh;
  ros::Subscriber sub = nh.subscribe("topic_name", 10, callbackFunction);

  ros::spin();  // 设置循环调用回调函数

  return 0;
}
