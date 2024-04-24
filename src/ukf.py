#!/usr/bin/env python3
import math
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
R = np.diag([0.01, 0.01, 0.01])
Q = np.array([[0.001, 0, 0, 0, 0, 0],
			[0, 0.001, 0, 0, 0, 0],
			[0, 0, 0.001, 0, 0, 0],
			[0, 0, 0, 100, 0, 0],
			[0, 0, 0, 0, 100, 0],
			[0, 0, 0, 0, 0, 100]])
# Don't use Q_discrete_white_noise, it is not even

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
first_ukf_frame = True
pose_last = 0
time_last = 0
pose_now = 0
time_now = 0
dt = 0
pub_nv = rospy.Publisher('naive_vel', PointStamped, queue_size = 1)
pub_kv = rospy.Publisher('kf_vel', PointStamped, queue_size = 1)
pub_ukv = rospy.Publisher('ukf_vel', PointStamped, queue_size = 1)
E_last = 0
E_now = 0
v_last = 0
v_now = 0
pub_drag = rospy.Publisher('drag_coefficient', PointStamped, queue_size = 1)
count = 0
t_last = 0
t_now = 0
pub_ukx = rospy.Publisher('ukf_pos', PointStamped, queue_size = 1)
pub_kx = rospy.Publisher('kf_pos', PointStamped, queue_size = 1)

def saturate(x, lim):
	if x > lim:
		return lim
	elif x < -lim:
		return -lim
	else:
		return x

def callback(msg):
	global first_frame, pose_last, time_last, pose_now, time_now, kf, ukf, first_ukf_frame, E_last, E_now, v_last, v_now, count, t_last, t_now
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
	kf_output = get_kf_output(kf, pose_now, dt).tolist()
	kf_v = kf_output[3:6]
	kf_x = kf_output[0:3]
	# print(kf_v)
	ukf_output = get_ukf_output(ukf, pose_now, dt).tolist()
	ukf_v = ukf_output[3:6]
	ukf_x = ukf_output[0:3]

	n_v = PointStamped()
	n_v.header.stamp = rospy.Time.from_sec(time_now)
	n_v.point.x = saturate(naive_vel[0], 5)
	n_v.point.y = saturate(naive_vel[1], 5)
	n_v.point.z = saturate(naive_vel[2], 5)
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

	uk_x = PointStamped()
	uk_x.header.stamp = n_v.header.stamp
	uk_x.point.x = ukf_x[0]
	uk_x.point.y = ukf_x[1]
	uk_x.point.z = ukf_x[2]
	pub_ukx.publish(uk_x)

	k_x = PointStamped()
	k_x.header.stamp = n_v.header.stamp
	k_x.point.x = kf_x[0]
	k_x.point.y = kf_x[1]
	k_x.point.z = kf_x[2]
	pub_kx.publish(k_x)

	pose_last = pose_now
	time_last = time_now

	if first_ukf_frame == True:
		first_ukf_frame = False
		v_last = ukf_v
		E_last = 1 / 2 * (v_last[0] ** 2 + v_last[1] ** 2 + v_last[2] ** 2) + 9.81 * pose_now[2]
		t_last = time_now
		return
	
	if count < 10:
		count += 1
		return
	
	count = 0
	t_now = time_now
	dt_long = t_now - t_last
	
	v_now = ukf_v
	E_now = 1 / 2 * (v_now[0] ** 2 + v_now[1] ** 2 + v_now[2] ** 2) + 9.81 * pose_now[2]
	v_avg = [i/2 + j/2 for i, j in zip(v_last, v_now)]
	drag_coefficient = (E_last - E_now) / math.sqrt(v_avg[0] ** 2 + v_avg[1] ** 2 + v_avg[2] ** 2) ** 3 / dt_long
	
	drag = PointStamped()
	drag.header.stamp = n_v.header.stamp
	drag.point.x = E_now #saturate(drag_coefficient, 2)
	drag.point.y = E_now - E_last #saturate(drag_coefficient, 2)
	drag.point.z = saturate(drag_coefficient, 2)
	pub_drag.publish(drag)

	v_last = v_now
	E_last = E_now
	





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
