import numpy as np

def get_ball_traj(p00: np.ndarray, v00: np.ndarray, target_height: float, kd_est: float, ddt: float = 0.0002):
    """
    :param p00: 球初始位置
    :param v00: 球初始速度
    :param target_height: 给定的预测的下限高度
    :param kd_est: 估计的空气阻力系数
    :param ddt: 模拟步长
    :return: 球的位置轨迹(三次式)参数及速度轨迹(三次式)参数【共六条轨迹】
    """
    '''
    如果已知空气阻力系数，那么前面可以省略求取空气阻力系数部分，只需要根据动捕数据求解球速
    '''

    a = np.array([0, 0, -9.81])
    # 求取轨迹
    ball_pose = []
    ball_vel = []
    t = []
    ball_pose.append(p00)
    ball_vel.append(v00)
    t_inl = 0.0
    t.append(t_inl)
    # 根据公式迭代出球在空间的球姿，限制条件为球在指定的高度上方并且速度较小
    while (ball_pose[-1][2] >= target_height) and (abs(ball_vel[-1][2]) <= 10):
        p = ball_pose[-1] + ball_vel[-1]*ddt + 0.5*a*ddt**2
        # p = ball_pose[-1] + ball_vel[-1]*ddt
        ball_pose.append(p)
        v = ball_vel[-1] - kd_est * np.linalg.norm(ball_vel[-1])*ddt*ball_vel[-1] + a*ddt
        ball_vel.append(v)
        t_inl += ddt
        t.append(t_inl)
    '''
    此时已大致完成了球的轨迹的求取，由于均为散点，因此多一步多项式拟合
    但也可以直接根据指定的碰撞高度，找到对应的相近的ball_pose及对应的时间，同样可以得到所需要的碰撞点的数据
    ————碰撞点坐标(x,y,z)、碰撞点球速(vx,vy,vz)
    '''
    target_ball_pose = ball_pose[-1]
    target_ball_vel = ball_vel[-1]
    # pose_poly_par = []
    # vel_poly_par = []
    # for i in range(3):
    #     pose_poly = np.polyfit(t, ball_pose[:, i], 3)
    #     pose_poly_par.append(pose_poly)
    #     vel_poly = np.polyfit(t, ball_vel[:, i], 7)
    #     vel_poly_par.append(vel_poly)
    # return np.stack(pose_poly_par), np.stack(vel_poly_par)

    # return target_ball_pose, target_ball_vel, t_inl
    return ball_pose, ball_vel, t

if __name__ == '__main__':
    ball_pose, ball_vel, t = get_ball_traj(np.array([]))