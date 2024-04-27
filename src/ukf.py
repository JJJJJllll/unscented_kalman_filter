#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import PoseStamped, PointStamped
import numpy as np
from filterpy.kalman import KalmanFilter as KF

'''
[[]]  [ , ]
1x1-R 1x3-H 3x1-B/x 3x3-F
'''
def saturate(x, lim):
	if x > lim:
		return lim
	elif x < -lim:
		return -lim
	else:
		return x

class optionalEKF(KF):
	def __init__(self,
			  	name: str):
		self.name = name
		if self.name == "hv":
			super().__init__(dim_x=2, dim_z=1, dim_u=0)
			self.H = np.array([[1, 0]])
			self.R = np.array([[0.05]])
			self.Q = np.diag([0.01, 100])
		elif self.name == "hvk":
			super().__init__(dim_x=3, dim_z=1, dim_u=1)
			self.H = np.array([[1, 0, 0]])
			self.R = np.array([[0.05]])
			self.Q = np.diag([0.01, 1000, 1000])
		elif self.name == "k":
			super().__init__(dim_x=1, dim_z=1, dim_u=0)
			self.H = np.array([[1]])
			self.Q = np.diag([0.00001]) # define k=k, in other words as smooth as possible
			self.F = np.array([[1]])
	
	def load_x0(self,
			 	x0: np.array):
		self.x = x0
		
	def linearize_FB(self, dt):
		self.dt = dt
		if self.name == "hv":
			self.F = np.array([[1, dt],
				   			[0, 1]])
			self.B = np.array([[0], [dt]])
		elif self.name == "hvk":
			if self.x[1, 0] > 0:
				self.F = np.array([[1, dt - self.x[2, 0] * self.x[1, 0] * dt ** 2, - 0.5 * self.x[1, 0] ** 2 * dt ** 2],
								[0, 1 - 2 * self.x[2, 0] * self.x[1, 0] * dt, - self.x[1, 0] ** 2 * dt],
								[0, 0, 1]])
			elif self.x[1, 0] == 0:
				self.F = np.array([[1, dt, 0],
								[0, 0, 0],
								[0, 0, 1]])
			else:
				self.F = np.array([[1, dt + self.x[2, 0] * self.x[1, 0] * dt ** 2, 0.5 * self.x[1, 0] ** 2 * dt ** 2],
								[0, 1 + 2 * self.x[2, 0] * self.x[1, 0] * dt, self.x[1, 0] ** 2 * dt],
								[0, 0, 1]])
			self.B = np.array([[0], [dt], [0]])

	def dynamic_R(self, R):
		self.R = R

	def predict_update(self, pos, u=None):
		if u is not None:
			self.predict(u)
		else:
			self.predict()
		self.update(pos)


first_frame = True
t_last = 0
z_last = 0
pub_nv = rospy.Publisher('nv', PointStamped, queue_size = 1)

kf_vel = optionalEKF("hv")
pub_vel = rospy.Publisher('kf_vel', PointStamped, queue_size = 1)

kf_drag = optionalEKF(name="hvk")
pub_drag = rospy.Publisher('kf_drag', PointStamped, queue_size = 1)

count = 0
WARM_UP = 12

E_last = 0
v_last = 0

kf_k = optionalEKF(name="k")
pub_k = rospy.Publisher('kf_k', PointStamped, queue_size = 1)

P = 1
k_last = 0

def callback(msg):
	global first_frame, t_last, z_last, kf_vel, kf_drag, count, WARM_UP, E_last, kf_k, v_last, P, k_last
	if first_frame == True:
		first_frame = False
		t_last = msg.header.stamp.to_sec()
		z_last = msg.pose.position.z

		kf_vel.load_x0(x0=np.array([[z_last], [0]]))
		kf_drag.load_x0(x0=np.array([[z_last], [0], [0.4]]))
		return
	
	t_now = msg.header.stamp.to_sec()
	dt = t_now - t_last
	z = msg.pose.position.z
	
	kf_vel.linearize_FB(dt)
	kf_vel.predict_update(pos=z)
 
	kf_drag.linearize_FB(dt)
	kf_drag.predict_update(pos=z, u=-9.8)
	
	vel = PointStamped()
	vel.header.stamp = rospy.Time.from_sec(t_now)
	vel.point.x = kf_vel.x[0, 0] # h
	vel.point.y = kf_vel.x[1, 0] # hdot
	pub_vel.publish(vel)

	nv_pub = PointStamped()
	nv_pub.header.stamp = vel.header.stamp
	nv_pub.point.z = (z - z_last) / dt
	pub_nv.publish(nv_pub)

	drag = PointStamped()
	drag.header.stamp = vel.header.stamp
	drag.point.x = kf_drag.x[0, 0]
	drag.point.y = kf_drag.x[1, 0]
	drag.point.z = kf_drag.x[2, 0]
	pub_drag.publish(drag)
	
	t_last = t_now
	z_last = z

	count += 1
	if count < WARM_UP:
		v_last = kf_vel.x[1, 0]
		E_last = 9.8 * z + 0.5 * v_last ** 2
		return
	v_now = kf_vel.x[1, 0]
	E_now = 9.8 * z + 0.5 * v_now ** 2
	v_avg = abs(v_now + v_last) / 2
	k = abs(E_last - E_now) / dt / v_avg ** 3
	rospy.loginfo(f"calc k is {k}")
	if count == WARM_UP:
		kf_k.load_x0(x0=np.array([[k]]))
		k_last = k
	else:
		R = 1 / v_avg ** 3
		C = P / (P + R)
		k_now = k_last + C * (k - k_last)
		P = P * R / (P + R)
		k_last = k_now

		kf_k.dynamic_R(R=np.array([[R]]))
		kf_k.predict_update(pos=k)

		k_pub = PointStamped()
		k_pub.header.stamp = vel.header.stamp
		k_pub.point.x = k
		k_pub.point.y = kf_k.x
		k_pub.point.z = k_now
		pub_k.publish(k_pub)

	E_last = E_now
	v_last = v_now
	

	
	 

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
