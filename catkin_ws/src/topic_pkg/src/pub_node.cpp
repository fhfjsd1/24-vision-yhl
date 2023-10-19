#include "ros/ros.h"
#include "std_msgs/String.h"

int main(int argc, char **argv) {
  ros::init(argc, argv, "pub_node_name");
  ros::NodeHandle nh;
  ros::Publisher pub = nh.advertise<std_msgs::String>("topic_name", 10);
  std_msgs::String msg;
  int count = 0;
  ros::Rate rate(1);  // 设置循环频率为1Hz

  while (ros::ok()) {
    msg.data = std::to_string(count);
    pub.publish(msg);
    rate.sleep();
    // ROS_INFO("写出的数据:%s", msg.data.c_str());  // 打印日志信息
    ROS_INFO("msg = %s, msg.data = %s", msg.data.c_str(), msg.data.c_str());
    count++;
  }

  return 0;
}
