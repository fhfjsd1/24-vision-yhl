#! /usr/bin/env python3
#coding:utf-8
import rospy
from std_msgs.msg import String

def CallBack_Function(msg):
    rospy.loginfo("I heard:%s",msg.data)

if __name__ == "__main__":
    # 初始化 ROS 节点:命名(唯一)
    rospy.init_node("sub_node_name")
    # 实例化 订阅者 对象
    sub = rospy.Subscriber("topic_name",String,CallBack_Function,queue_size=10)
    # CallBack_Function 处理订阅的消息(回调函数) 
    # 只用填函数名，ros会自动将订阅到的消息作为其入口参数传给回调函数
    # 设置循环调用回调函数
    rospy.spin()
