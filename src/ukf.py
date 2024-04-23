#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import PoseStamped, PointStamped
import numpy as np
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import MerweScaledSigmaPoints

dt = 1 / 120
x_initial = np.array([0., 0., 0., 0., 0., 0.])
F = np.array([[1, 0, 0, dt, 0, 0],
			[0, 1, 0, 0, dt, 0],
			[0, 0, 1, 0, 0, dt],
			[0, 0, 0, 1, 0, 0],
			[0, 0, 0, 0, 1, 0],
			[0, 0, 0, 0, 0, 1]])
H = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
			[0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
			[0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
R = np.eye(3) * 0.001;
R = np.diag([0.005**2, 0.005**2, 0.005**2])
Q = np.array([[0.001, 0, 0, 0, 0, 0],
			[0, 0.001, 0, 0, 0, 0],
			[0, 0, 0.001, 0, 0, 0],
			[0, 0, 0, 10, 0, 0],
			[0, 0, 0, 0, 10, 0],
			[0, 0, 0, 0, 0, 10]])
Q[0:3, 0:3] = Q_discrete_white_noise(3, dt=dt, var=0.)
Q[3:6, 3:6] = Q_discrete_white_noise(3, dt=dt, var=10)

def fx(x, dt):
	F = np.array([[1, 0, 0, dt, 0, 0],
                  [0, 1, 0, 0, dt, 0],
                  [0, 0, 1, 0, 0, dt],
                  [0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 1]])
	return F @ x

def hx(x):
	return x[[0, 1, 2]]
	
def create_ukf():
	global x_initial, R, Q
	sigmas = MerweScaledSigmaPoints(6, alpha=0.0001, beta=2., kappa=-3.)
	ukf = UKF(dim_x=6, dim_z=3, dt=dt, 
		   hx=hx, fx=fx, points=sigmas)
	ukf.x = x_initial
	ukf.R = R
	ukf.Q = Q
	return ukf

def get_ukf_output(ukf, pose, dt):
	ukf.dt = dt
	ukf.predict()
	ukf.update(pose)
	return ukf.x.copy()

def create_kf():
	global x_initial, F, H, R, Q
	kf = KalmanFilter(dim_x=6, dim_z=3)
	kf.x = x_initial
	kf.F = F
	kf.H = H
	kf.P = np.eye(6) * 10
	kf.R = R
	kf.Q = Q
	return kf

def get_kf_output(kf, pose, dt): 
	kf.dt = dt
	kf.predict()
	kf.update(pose)
	return kf.x.copy()

ukf = create_ukf()
kf = create_kf()

first_frame = True
pose_last = 0
time_last = 0
pose_now = 0
time_now = 0
dt = 0
pub_nv = rospy.Publisher('naive_vel', PointStamped, queue_size = 1)
pub_kv = rospy.Publisher('kf_vel', PointStamped, queue_size = 1)
pub_ukv = rospy.Publisher('ukf_vel', PointStamped, queue_size = 1)

def callback(msg):
	global first_frame, pose_last, time_last, pose_now, time_now, kf, ukf
	if first_frame == True:
		first_frame = False
		pose_last = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
		time_last = msg.header.stamp.to_sec()
		print(pose_last, time_last)
		return
	
	pose_now = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
	time_now = msg.header.stamp.to_sec()
	
	pose_diff = [i - j for i, j in zip(pose_now, pose_last)]
	dt = time_now - time_last
	if dt < 0.001:
		print(dt, "*" * 20)
	elif dt > 0.01: 
		print(dt, "/" * 20)
	else:
		print(dt)

	naive_vel = [k / (time_now - time_last) for k in pose_diff]
	# print(naive_vel)
	kf_v = get_kf_output(kf, pose_now, dt).tolist()[3:6]
	# print(kf_v)
	ukf_v = get_ukf_output(ukf, pose_now, dt).tolist()[3:6]

	n_v = PointStamped()
	n_v.header.stamp = rospy.Time.from_sec(time_now)
	n_v.point.x = naive_vel[0]
	n_v.point.y = naive_vel[1]
	n_v.point.z = naive_vel[2]
	pub_nv.publish(n_v)

	k_v = PointStamped()
	k_v.header.stamp = n_v.header.stamp
	k_v.point.x = kf_v[0]
	k_v.point.y = kf_v[1]
	k_v.point.z = kf_v[2]
	pub_kv.publish(k_v)

	uk_v = PointStamped()
	uk_v.header.stamp = n_v.header.stamp
	uk_v.point.x = ukf_v[0]
	uk_v.point.y = ukf_v[1]
	uk_v.point.z = ukf_v[2]
	pub_ukv.publish(uk_v)

	pose_last = pose_now
	time_last = time_now




def main():
	rospy.init_node("ukf", anonymous=True)
	rospy.loginfo("ukf node init")

	pose_name = "/natnet_ros/ball/pose"
	rospy.Subscriber(pose_name, PoseStamped, callback)
	rospy.loginfo(f"ukf node is subscribing {pose_name}")

	
	rospy.spin()


if __name__ == '__main__':
	try:
		main()
	except rospy.ROSInterruptException:
		print("Failed to start ukf!")
		pass
