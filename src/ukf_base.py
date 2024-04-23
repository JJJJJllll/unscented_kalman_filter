#!/usr/bin/env python3
import numpy as np
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import MerweScaledSigmaPoints

def f_x(x, dt):
    """
    :param x: 球的质心位置及速度([x, x', y, y', z, z'])
    :param dt: 时间间隔
    :return: 球的运动矩阵
    """
    F = np.array([[1, 0, 0, dt, 0, 0],
                  [0, 1, 0, 0, dt, 0],
                  [0, 0, 1, 0, 0, dt],
                  [0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 1]])
    return F @ x

def h_cv(x):
    return x[[0, 1, 2]]

def get_ukf_data(ukf, msg):
    ukf.predict()
    ukf.update(msg)
    data_ukf = ukf.x.copy()
    return data_ukf


def make_ukf(dt: float):
    sigmas = MerweScaledSigmaPoints(6, alpha=0.0001, beta=2., kappa=-3.)
    ukf_p = UKF(dim_x=6, dim_z=3, fx=f_x,
                hx=h_cv, dt=dt, points=sigmas)
    # 初始化UKF状态和协方差矩阵
    ukf_p.x = np.array([0., 0., 0., 0., 0., 0.])
    # 假设测量误差为0.005m
    ukf_p.R = np.diag([0.005**2, 0.005**2, 0.005**2])
    # 过程噪声————只作用于速度项
    ukf_p.Q[0:3, 0:3] = Q_discrete_white_noise(3, dt=dt, var=0.)
    ukf_p.Q[3:6, 3:6] = Q_discrete_white_noise(3, dt=dt, var=10)
    return ukf_p
 
def make_kf(dt: float):
    kf = KalmanFilter (dim_x=6,
                       dim_z=3)
    kf.x = np.array([0.0, 0.0, 1.4, 0.0, 0.0, 6.0])
    kf.F = np.array([[1, 0, 0, dt, 0, 0],
                    [0, 1, 0, 0, dt, 0],
                    [0, 0, 1, 0, 0, dt],
                    [0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 1]])
    kf.H = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                     [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                     [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
    kf.P = np.eye(6) * 10
    kf.R = np.eye(3) * 0.001
    kf.Q = np.array([[0.001, 0, 0, 0, 0, 0],
                    [0, 0.001, 0, 0, 0, 0],
                    [0, 0, 0.001, 0, 0, 0],
                    [0, 0, 0, 10, 0, 0],
                    [0, 0, 0, 0, 10, 0],
                    [0, 0, 0, 0, 0, 10]])
    return kf

def get_kf_x(kf, pose):    
    kf.predict()
    kf.update(pose)
    
    return kf.x.copy()
