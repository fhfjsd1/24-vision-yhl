#! /usr/bin/env python3
#coding:utf-8


import rospy
from std_msgs.msg import String

if __name__ == "__main__":
    rospy.init_node("pub_node_name")    
    # 实例化 发布者 对象
    pub = rospy.Publisher("topic_name",String,queue_size=10)
    # 组织被发布的数据
    msg = String()  #创建 msg 对象
    count = 0  #计数器 
    # 设置循环频率
    rate = rospy.Rate(1)    #Hz
    while not rospy.is_shutdown():
        msg.data = str(count)
        pub.publish(msg)
        rate.sleep()
        # rospy.loginfo("写出的数据:%s",msg.data)  #打印日志信息
        print("msg = ", msg, "msg.data = ", msg.data)
        print(type(msg), type(msg.data)) 
        count += 1
