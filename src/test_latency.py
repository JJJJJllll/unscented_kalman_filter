#!/usr/bin/env python3
# 上面这行用来给权限，否则不能import rosmsg
import rospy
from geometry_msgs.msg import PoseStamped
import time

pub = rospy.Publisher('a', PoseStamped, queue_size=1)

rospy.init_node('test_latency')

r = rospy.Rate(100) # 10hz

while not rospy.is_shutdown():
    msg = PoseStamped()
    t_now = time.time()
    msg.header.stamp = rospy.Time.now()
    print(1000 * (msg.header.stamp.to_time() - t_now) ,'ms')
    # 上面两行说明了time.time()和rospy.Time.now()时间是一致的，只差10us（两行程序毕竟不是同时被执行的）
    msg.pose.position.x = 1
    msg.pose.position.y = 2
    msg.pose.position.z = 3
    msg.pose.orientation.x = 4
    msg.pose.orientation.y = 5
    msg.pose.orientation.z = 6
    msg.pose.orientation.w = 7
    pub.publish(msg)
    r.sleep()