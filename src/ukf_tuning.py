#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import PoseStamped
import numpy as np
from filterpy.kalman import KalmanFilter as KF
from filterpy.kalman import UnscentedKalmanFilter as UKF
import time
import sys # 用来退出
import signal # 用来退出程序
import matplotlib.pyplot as plt # 用来画方差和残差

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
		elif self.name == "x3v3":
			super().__init__(dim_x=6, dim_z=3, dim_u=0)
			self.H = np.zeros([3,6])
			np.fill_diagonal(self.H, 1)
			self.R = np.diag([0.02, 0.02, 0.02])
			self.Q = np.diag([0.01, 0.01, 0.01, 100, 100, 100])
	
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
		elif self.name == "x3v3":
			self.F = np.eye(6)
			self.F[0:3, 3:6] = np.eye(3) * dt

	def dynamic_R(self, R):
		self.R = R

	def predict_update(self, pos, u=None):
		if u is not None:
			self.predict(u)
		else:
			self.predict()
		self.update(pos)


class onlinePredict():
	def __init__(self):
		'''for estimator'''
		self.first_frame = True
		self.t_last = 0
		self.t_now = 0
		self.z_last = 0
		self.z_now = 0
		self.pose_last = 0
		self.pose_now = 0
		self.count = 0
		self.WARM_UP = 12
		self.WARM_UP_OK = False
		self.E_last = 0
		self.v_last = 0
		self.v_now = 0
		self.P = 1
		self.k_last = 0
		self.k_now = 0

		self.x_subscriber = rospy.Subscriber("/natnet_ros/ball/pose", PoseStamped, self.estimator)
		
		self.kf_vel = optionalEKF(name="hv")
		self.kf_x3v3 = optionalEKF(name="x3v3")
		self.ekf_drag = optionalEKF(name="hvk")
		self.ekf_k = optionalEKF(name="k")
	
		self.pub_nv = rospy.Publisher('nv', PointStamped, queue_size = 1)
		self.pub_vel = rospy.Publisher('kf_vel', PointStamped, queue_size = 1)
		self.pub_drag = rospy.Publisher('ekf_drag', PointStamped, queue_size = 1)
		self.pub_k = rospy.Publisher('ekf_rls_k', PointStamped, queue_size = 1)
		self.pub_x3v3 = rospy.Publisher('kf_x3v3', PoseStamped, queue_size = 1)
		rospy.loginfo("subscriber /natnet_ros/ball/pose init, publisher nv, kf_vel, kf_x3v3, eekf_drag, ekf_k init")

		'''for predictor'''
		self.ball_pose = []
		self.ball_vel = []
		self.ball_t = []
		self.g = - 9.8
		self.first_predictor = True
		self.pub_h = rospy.Publisher('pred_h', PointStamped, queue_size=1)

	def estimator(self, msg):
		if self.first_frame == True:
			self.first_frame = False
			self.t_last = msg.header.stamp.to_sec()
			self.z_last = msg.pose.position.z
			self.pose_last = np.array([[msg.pose.position.x], [msg.pose.position.y], [msg.pose.position.z]])

			self.kf_vel.load_x0(x0=np.array([[self.z_last], [0]]))
			self.kf_x3v3.load_x0(x0=np.concatenate((self.pose_last, np.array([[0], [0], [0]])), axis=0))
			self.ekf_drag.load_x0(x0=np.array([[self.z_last], [0], [0.4]]))
			return
		
		self.t_now = msg.header.stamp.to_sec()
		dt = self.t_now - self.t_last
		self.z_now = msg.pose.position.z
		self.pose_now = np.array([[msg.pose.position.x], [msg.pose.position.y], [msg.pose.position.z]])
		
		self.kf_vel.linearize_FB(dt)
		self.kf_vel.predict_update(pos=self.z_now)

		self.kf_x3v3.linearize_FB(dt)
		self.kf_x3v3.predict_update(pos=self.pose_now)
	
		self.ekf_drag.linearize_FB(dt)
		self.ekf_drag.predict_update(pos=self.z_now, u=-9.8)
		
		vel = PointStamped()
		vel.header.stamp = rospy.Time.from_sec(self.t_now)
		vel.point.x = self.kf_vel.x[0, 0] # h
		vel.point.y = self.kf_vel.x[1, 0] # hdot
		self.pub_vel.publish(vel)

		x3v3 = PoseStamped()
		x3v3.header.stamp = vel.header.stamp
		x3v3.pose.position.x = self.kf_x3v3.x[0, 0]
		x3v3.pose.position.y = self.kf_x3v3.x[1, 0]
		x3v3.pose.position.z = self.kf_x3v3.x[2, 0]
		x3v3.pose.orientation.x = self.kf_x3v3.x[3, 0]
		x3v3.pose.orientation.y = self.kf_x3v3.x[4, 0]
		x3v3.pose.orientation.z = self.kf_x3v3.x[5, 0]
		self.pub_x3v3.publish(x3v3)

		nv_pub = PointStamped()
		nv_pub.header.stamp = vel.header.stamp
		nv_pub.point.z = (self.z_now - self.z_last) / dt
		self.pub_nv.publish(nv_pub)

		drag = PointStamped()
		drag.header.stamp = vel.header.stamp
		drag.point.x = self.ekf_drag.x[0, 0]
		drag.point.y = self.ekf_drag.x[1, 0]
		drag.point.z = self.ekf_drag.x[2, 0]
		self.pub_drag.publish(drag)
		
		self.t_last = self.t_now
		self.z_last = self.z_now
		self.pose_last = self.pose_now

		self.count += 1
		if self.count < self.WARM_UP:
			self.v_last = self.kf_vel.x[1, 0]
			self.E_last = 9.8 * self.z_now + 0.5 * self.v_last ** 2
			return
		self.WARM_UP_OK = True
		self.v_now = self.kf_vel.x[1, 0]
		E_now = 9.8 * self.z_now + 0.5 * self.v_now ** 2
		v_avg = abs(self.v_now + self.v_last) / 2
		k = abs(self.E_last - E_now) / dt / v_avg ** 3

		if self.count == self.WARM_UP:
			self.ekf_k.load_x0(x0=np.array([[k]]))
			self.k_last = k
		else:
			R = 1 / v_avg ** 3
			# R = np.var(np.array([self.v_now, self.v_last])) / v_avg ** 3
			C = self.P / (self.P + R)
			self.k_now = self.k_last + C * (k - self.k_last)
			self.P = self.P * R / (self.P + R)
			self.k_last = self.k_now

			self.ekf_k.dynamic_R(R=np.array([[R]]))
			self.ekf_k.predict_update(pos=k)

			k_pub = PointStamped()
			k_pub.header.stamp = vel.header.stamp
			k_pub.point.x = k
			k_pub.point.y = self.ekf_k.x
			k_pub.point.z = self.k_now
			self.pub_k.publish(k_pub)

		self.E_last = E_now
		self.v_last = self.v_now

	def predictor(self, timeevent):
		if not self.WARM_UP_OK :
			return
		if self.first_predictor == True:
			self.first_predictor = False
			self.ball_pose.append(self.z_now)
			self.ball_vel.append(self.kf_vel.x[1, 0])
			self.ball_t.append(self.t_now)
		if self.ball_t[-1] - self.ball_t[0] > 1:
			return 	
		pred_v_now = self.ball_vel[-1]
		adrag = - self.k_now * abs(pred_v_now) * pred_v_now
		# adrag = - 0.1 * abs(pred_v_now) * pred_v_now  
		a = adrag + self.g
		# if pred_v_now < 0:
		# 	a = -2 + self.g
		# elif pred_v_now > 0:
		# 	a = 2 + self.g	
		# else:
		# 	a = self.g
		# if self.v_now > 0:
		# 	a = -0.5 + self.g
		# elif self.v_now < 0:
		# 	a = 0.8 + self.g	
		# else:
		# 	a = self.g
		# a = self.g

		dt = 0.01
		self.ball_pose.append(self.ball_pose[-1] + self.ball_vel[-1]*dt + 0.5*a*dt**2)
		self.ball_vel.append(self.ball_vel[-1] + a * dt)
		self.ball_t.append(self.ball_t[-1] + dt)

		# if self.ball_t[-1] - self.ball_t[0] < 0.2:
		# 	self.ball_vel[-1] = self.v_now
		# 	self.ball_pose[-1] = self.z_now

		h = PointStamped()
		h.header.stamp = rospy.Time.from_sec(self.ball_t[-1])
		h.point.x = self.ball_pose[-1] # h
		h.point.y = self.ball_vel[-1] # hdot
		self.pub_h.publish(h)

		# while (ball_pose[-1][2] >= target_height) and (abs(ball_vel[-1][2]) <= 10):
		# 	p = ball_pose[-1] + ball_vel[-1]*ddt + 0.5*a*ddt**2
		# 	# p = ball_pose[-1] + ball_vel[-1]*ddt
		# 	ball_pose.append(p)
		# 	v = ball_vel[-1] - kd_est * np.linalg.norm(ball_vel[-1])*ddt*ball_vel[-1] + a*ddt
		# 	ball_vel.append(v)
		# 	t_inl += ddt
		# 	# t.append(t_inl)

'''
Feature:
1. 测试natnet包到filter包的延迟
2. 用natnet的时间——物理量出来那一刻的时间、而不用当前时间
3. 测试filter耗时
4. 两个kf对比
新：
5. 用方差和残差定量分析滤波器效果
6. 退出rosnode开始分析数据
'''
def set_labels(title=None, x=None, y=None):
    """ helps make code in book shorter. Optional set title, xlabel and ylabel
    """
    if x is not None:
        plt.xlabel(x)
    if y is not None:
        plt.ylabel(y)
    if title is not None:
        plt.title(title)

def plot_residual_limits(Ps, stds=1.):
    """ plots standard deviation given in Ps as a yellow shaded region. One std
    by default, use stds for a different choice (e.g. stds=3 for 3 standard
    deviations.
    """

    std = np.sqrt(Ps) * stds

    plt.plot(-std, color='k', ls=':', lw=2)
    plt.plot(std, color='k', ls=':', lw=2)
    plt.fill_between(range(len(std)), -std, std,
                 facecolor='#ffff00', alpha=0.3)
	
def plot_residuals(res, P, title, y_label, stds=1):
	plt.plot(res)
	plot_residual_limits(P, stds)
	set_labels(title, 'time (sec)', y_label)

def plot_multiple_residuals(filters):
	for n, filter in enumerate(filters):
		plt.figure()
		plot_residuals(filter.zRes, filter.zP, str(n)+'height', 'meters', stds=1)
	plt.show()

def signal_handler(signal,frame):
	print('You pressed Ctrl + C!')
	rospy.signal_shutdown('test')

class filterTuning():
	
	def __init__(self, type: str, num: int, ):
		natnet_sub = rospy.Subscriber("/natnet_ros/ball/pose", PoseStamped, self.natnet_callback)
		self.filter_pub = rospy.Publisher(type+str(num), PoseStamped, queue_size = 1) # kf0, ukf0
		self.lastTime = 0.
		self.type, self.num = type, num
		self.zRes, self.zP = [], [] # 用来看滤波器性能
		self.vzRes, self.vzP = [], [] # 用来看滤波器性能

		print(type, num)
		if type == 'kf' or type == 'KF':
			self.filter = KF(dim_x=6, dim_z=3, dim_u=0)
			self.filter.P = 10. * np.eye(6)
			self.filter.H = np.concatenate((np.diag([1., 1., 1.]), np.zeros((3,3))), axis=1) 
			self.filter.R = np.diag([0.001, 0.001, 0.001])
			self.filter.Q = np.diag([0.001, 0.001, 0.001, 10, 10, 10]) # 速度的过程噪声大，因为更新方法不对

			if num == 1:
				self.filter.P = np.diag([4e-8, 4e-8, 4e-8, 9, 9, 9])

				self.filter.R = 4e-8 * np.eye(3)

				self.filter.Q = np.diag([0., 0., 0., 4e-4, 4e-4, 4e-4])
				
			print('输出方程H', self.filter.H.shape, '\n', self.filter.H)
			print('观测误差R', self.filter.R.shape, '\n', self.filter.R)
			print('协方差矩阵P', self.filter.P.shape, '\n', self.filter.P)
			print('过程噪声Q', self.filter.Q.shape, '\n', self.filter.Q)
		# elif type == 'ekf' or type == 'EKF':
		elif type == 'ukf' or type == 'UKF':
			self.filter = UKF(dim_x=6, dim_z=3, dim_u=0)
		else:
			rospy.logerr('invalid filter type')
			sys.exit(0)
		
	
	def natnet_callback(self, msg):
		pos = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
		SendTime = msg.header.stamp.to_time()
		# RecvTime = time.time()
		# print('RecvTime - SendTime%6.2fms, SendTime%10f, RecvTime%10f %10f' %(1000 * (RecvTime - SendTime), SendTime, RecvTime, rospy.Time.now().to_time()) ) # 测试natnet包到filter包的延迟
		if self.lastTime == 0.:
			self.lastTime = SendTime # 用真实物理量发生的时间 SendTime
			self.filter.x = np.concatenate((pos, np.zeros((3,))), axis=0)
			print('状态x', self.filter.x.shape, self.filter.x)
			print('观测z', pos.shape, pos)
			return
		dynamicDT = SendTime - self.lastTime
		self.lastTime = SendTime

		if self.type == 'kf' or self.type == 'KF': 
			self.filter.F = np.eye(6)
			self.filter.F[0:3, 3:6] = dynamicDT * np.eye(3)
		self.filter.predict()
		self.filter.update(pos)
		self.zRes.append(self.filter.y[2]) # 用来定量评估滤波器
		self.zP.append(self.filter.P[2,2])
		# self.vzRes.append(self.filter.y[5])
		# self.vzP.append(self.filter.P[5,5])

		msg = PoseStamped()
		msg.header.stamp = rospy.Time.from_sec(SendTime) # 用真实物理量发生的时间 SendTime 
		msg.pose.position.x = self.filter.x[0]
		msg.pose.position.y = self.filter.x[1]
		msg.pose.position.z = self.filter.x[2]
		msg.pose.orientation.x = self.filter.x[3]
		msg.pose.orientation.y = self.filter.x[4]
		msg.pose.orientation.z = self.filter.x[5]
		self.filter_pub.publish(msg)
		# print(self.type+str(self.num), 'postprocessing time - pub msg header time%6.2fms' % ((time.time() - SendTime) * 1000)) # 看处理耗时

		

		

if __name__ == '__main__':
	rospy.init_node("filter", anonymous=True)

	kf0 = filterTuning('kf', 0)

	kf1 = filterTuning('kf', 1)

	# rospy.Timer(rospy.Duration(1.0/100.0), oP.predictor)
	signal.signal(signal.SIGINT,signal_handler) # 随时退出程序

	rospy.spin()

	print('rosnode died but i still print here')

	# plot_residuals(kf0.zRes, kf0.zP, 'kf0_height', 'meters', stds=1)
	
	# plot_residuals(kf1.zRes, kf1.zP, 'kf1_height', 'meters', stds=1)

	plot_multiple_residuals((kf0,kf1))


