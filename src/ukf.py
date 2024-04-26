#!/usr/bin/env python3
import math
import rospy
from geometry_msgs.msg import PoseStamped, PointStamped
import numpy as np
from filterpy.kalman import KalmanFilter as KF

def get_linearized_F(dt: float, k:float, v):
	'''v: np.array(3,)'''
	vnorm = v.dot(v) ** 0.5
	return np.array([[1, 0, 0, dt - 0.5 * k * (vnorm ** 2 * v[0] ** 2 + vnorm), 0, 0, - 0.5 * vnorm * v[0] * dt ** 2],
					[0, 1, 0, 0, dt - 0.5 * k * (vnorm ** 2 * v[1] ** 2 + vnorm), 0, - 0.5 * vnorm * v[1] * dt ** 2],
					[0, 0, 1, 0, 0, dt - 0.5 * k * (vnorm ** 2 * v[2] ** 2 + vnorm), - 0.5 * vnorm * v[2] * dt ** 2],
					[0, 0, 0, 1 - k * dt * (vnorm ** 2 * v[0] ** 2 + vnorm), 0, 0, - vnorm * v[0] * dt],
					[0, 0, 0, 0, 1 - k * dt * (vnorm ** 2 * v[1] ** 2 + vnorm), 0, - vnorm * v[1] * dt],
					[0, 0, 0, 0, 0, 1 - k * dt * (vnorm ** 2 * v[2] ** 2 + vnorm), - vnorm * v[2] * dt],
					[0, 0, 0, 0, 0, 0, 1]])

def get_kf_output(kf, pose, dt):
	kf.dt = dt
	kf.F = get_linearized_F(dt, kf.x[6, 0], kf.x[3:6, 0])
	kf.B = np.array([[0], [0], [-9.8 * dt], [0], [0], [dt], [0]])
	kf.predict(u=-9.8)
	kf.update(pose)
	return kf.x.copy()

'''
[[]]  [ , ]
3x3-R 3x7-H 7x1-B/x 7x7-F
'''
kf = KF(dim_x=7, dim_z=3, dim_u=1)
x0, y0, z0, vx0, vy0, vz0, k0, dt, vnorm0 = 1.25, 0.26, 1.3, -0.2, 0.1, 4.0, 0.06, 1 / 120, math.sqrt(0.2**2 + 0.1**2+ 4.0**2)
kf.x = np.array([[x0], [y0], [z0], [vx0], [vy0], [vz0], [k0]])
kf.dt = dt
kf.F = get_linearized_F(dt=dt, k=k0, v=np.array([vx0, vy0, vz0]))
kf.H = np.zeros([3, 7])
kf.H[0:3, 0:3] = np.eye(3)
kf.R = np.eye(3) * 0.01
kf.Q = np.diag([0.01, 0.01, 0.01, 10000, 10000, 10000, 10000])
kf.B = np.array([[0], [0], [0], [0], [0], [dt], [0]])

first_frame = True
time_last = 0
pub_kf_x = rospy.Publisher('kf_x', PointStamped, queue_size = 1)
pub_kf_v = rospy.Publisher('kf_v', PointStamped, queue_size = 1)
pub_kf_drag = rospy.Publisher('kf_drag', PointStamped, queue_size = 1)

pose_last = 0
pub_nv = rospy.Publisher('nv', PointStamped, queue_size = 1)

def callback(msg):
	global first_frame, time_last, kf, pose_last, pose_last
	rospy.loginfo(f"x: {kf.x.shape}, F: {kf.F.shape}, H: {kf.H.shape}, R: {kf.R.shape}. Q: {kf.Q.shape}, B: {kf.B.shape}")
	if first_frame == True:
		first_frame = False
		time_last = msg.header.stamp.to_sec()
		pose_last = msg.pose.position
		return
	
	time_now = msg.header.stamp.to_sec()
	dt = time_now - time_last
	
	pose = msg.pose.position
	kf_output = get_kf_output(kf, np.array([[pose.x], [pose.y], [pose.z]]), dt)
	# rospy.loginfo(f"data: {kf_output.shape}")

	kf_pub_x = PointStamped()
	kf_pub_x.header.stamp = rospy.Time.from_sec(time_now)
	kf_pub_x.point.x = kf_output[0, 0]
	kf_pub_x.point.y = kf_output[1, 0]
	kf_pub_x.point.z = kf_output[2, 0]
	pub_kf_x.publish(kf_pub_x)
	
	kf_pub_v = PointStamped()
	kf_pub_v.header.stamp = rospy.Time.from_sec(time_now)
	kf_pub_v.point.x = kf_output[3, 0]
	kf_pub_v.point.y = kf_output[4, 0]
	kf_pub_v.point.z = kf_output[5, 0]
	pub_kf_v.publish(kf_pub_v)

	kf_pub_drag = PointStamped()
	kf_pub_drag.header.stamp = rospy.Time.from_sec(time_now)
	kf_pub_drag.point.x = kf_output[6, 0]
	kf_pub_drag.point.y = kf_output[6, 0]
	kf_pub_drag.point.z = kf_output[6, 0]
	pub_kf_drag.publish(kf_pub_drag)

	nv_pub = PointStamped()
	nv_pub.header.stamp = kf_pub_x.header.stamp
	nv_pub.point.x = (pose.x - pose_last.x) / dt
	nv_pub.point.y = (pose.y - pose_last.y) / dt
	nv_pub.point.z = (pose.z - pose_last.z) / dt
	pub_nv.publish(nv_pub)
	
	time_last = time_now
	pose_last = pose



def main():
	rospy.init_node("kf", anonymous=True)
	rospy.loginfo("kf node init")

	pose_name = "/natnet_ros/ball/pose"
	rospy.Subscriber(pose_name, PoseStamped, callback)
	rospy.loginfo(f"kf node is subscribing {pose_name}")

	
	rospy.spin()


if __name__ == '__main__':
	try:
		main()
	except rospy.ROSInterruptException:
		print("Failed to start kf!")
		pass
