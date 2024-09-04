#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import PoseStamped
import numpy as np
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import MerweScaledSigmaPoints

FIRSTFRAME, SECONDFRAME = True, True
FirstPose, pret, dt, ukf_filter = np.array([0.0,0.0,0.0]), 0, 0, 0

def ballPoseCallback(msg, ball_xv_pub):
	global FIRSTFRAME, SECONDFRAME, FirstPose, pret, dt, ukf_filter
	if FIRSTFRAME == True:
		# 第一帧记录位置就返回
		FIRSTFRAME, pret, FirstPose = False, msg.header.stamp.to_sec(), np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
		return
	elif SECONDFRAME == True:
		# 第二帧初始化滤波器
		SECONDFRAME, nowt, SecondPose  = False, msg.header.stamp.to_sec(), np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
		dt, pret = nowt - pret, nowt

		ukfInitValue = np.zeros(shape = (6,)) # 位置和速度穿插 x vx y vy z vz
		ukfInitValue[0::2], ukfInitValue[1::2] = SecondPose, (SecondPose - FirstPose) / dt # 分别为奇数项[0::2]和偶数项[1::2]赋值
		
		sigmas = MerweScaledSigmaPoints(6, alpha=0.5, beta=2., kappa=-3.) # alpha取0.1，beta取2.，kappa取3-n
		ukf_filter = UKF(dim_x = 6, dim_z = 3, dt = dt, hx = hx, fx = fx, points = sigmas) # 创建6维(x vx y vy z vz)滤波器、观测3维(x y z)

		ukf_filter.x, ukf_filter.R = ukfInitValue, np.diag([0.01, 0.01, 0.01]) # 误差0.1mm做标准差，平方得到方差
		
		ukf_filter.Q = np.zeros(shape=(6,6))
		# ukf_filter.Q[0:2, 0:2] = ukf_filter.Q[2:4, 2:4] = ukf_filter.Q[4:6, 4:6] = Q_discrete_white_noise(2, dt=dt, var=1.) # 三组协方差一致
		ukf_filter.Q[0:2, 0:2] = ukf_filter.Q[2:4, 2:4] = ukf_filter.Q[4:6, 4:6] = np.array([[0.01, 0], [0, 100]])
		return
	else:
		# 更新滤波器
		nowt, observation = msg.header.stamp.to_sec(), np.array([[msg.pose.position.x], [msg.pose.position.y], [msg.pose.position.z]])
		
		observation = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
		
		dt, pret = nowt - pret, nowt # 顺序不能乱、否则计算dt出错 240804 jsl
		ukf_filter.dt = dt # 变dt
		ukf_filter.predict()
		ukf_filter.update(observation)
		# 发布消息：pose对应x-z、orient对应vx-vz
		xv = PoseStamped()
		xv.header.stamp = msg.header.stamp
		xv.pose.position.x, xv.pose.orientation.x = ukf_filter.x[0], ukf_filter.x[1]
		xv.pose.position.y, xv.pose.orientation.y = ukf_filter.x[2], ukf_filter.x[3]
		xv.pose.position.z, xv.pose.orientation.z = ukf_filter.x[4], ukf_filter.x[5]
		ball_xv_pub.publish(xv)

def hx(x):
	# 2.取值：取奇数项[0::2]
	return x[0::2]

def fx(x, dt):
	# 3.变值：奇数项+偶数项 * dt
	x[0::2] += x[1::2] * dt
	return x

# def fx():

def main():
	rospy.init_node("ukf", anonymous=True)
	ball_xv_pub = rospy.Publisher('ukf_xv', PoseStamped, queue_size = 1)
	ball_sub = rospy.Subscriber("/natnet_ros/ball/pose", PoseStamped, ballPoseCallback, ball_xv_pub)
	
	rospy.loginfo("subscriber /natnet_ros/ball/pose init, publisher ukf_xv")

	rospy.spin()


if __name__ == '__main__':
	try:
		main()
	except rospy.ROSInterruptException:
		print("Failed to start ukf!")
		pass