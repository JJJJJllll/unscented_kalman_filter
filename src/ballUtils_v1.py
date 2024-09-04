import numpy as np
import time
'''
加入了纯粹数值积分，分别取dt=0.0002s, 0.002s, 0.02s看时间
'''

def ballModel(y, t, k0, k1, k2, g):
    # y = [x, y, z, vx, vy, vz]
    v = np.array([y[3], y[4], y[5]], dtype = float)
    # x'(t) = v(t)
    dxdt = v
    # v'(t) = - k0*sign(v) -k1*v - k2*v*norm(v) + [0;0;-9.8]
    dvdt = - k0*np.sign(v) - k1*v - k2*v*np.linalg.norm(v,ord=2) + g
    # dydt = [x', v'] = [v, a]
    dydt = np.concatenate((dxdt, dvdt), axis=0)
    return dydt

def ballPred(y0, targetHeight=1, output='full traj'):
    # constants
    k0, k1, k2, g = 0.01, 0.05, 0.05, np.array([0,0,-9.8], dtype = float)

    # generate a solution at 101 evenly spaced samples in the interval 0 <= t <= 1. 
    t = np.linspace(0, 1, 201)

    # Call odeint to generate the solution. 
    # pass parameters to odeint using the args argument.
    from scipy.integrate import odeint
    sol = odeint(ballModel, y0, t, args=(k0, k1, k2, g))

    # find target height
    targetIndex = np.where(sol[:,2][::-1] > targetHeight)[0][0] # 反转，找到大于targetHeight的首个元素位置

    if output == 'full traj':
        return sol, t # return full trajectory
    elif output == 'cutoff traj':
        return sol[0:-targetIndex,:], t[0:-targetIndex]
    elif output == 'xv t':
        return sol[-targetIndex,:], t[-targetIndex]
    elif output == 'x v t':
        return sol[-targetIndex,0:3], sol[-targetIndex,3:], t[-targetIndex]

def ballPlot(sol, t):
    # The solution is an array with shape (101, 6). 
    # The following code plots both components.
    import matplotlib.pyplot as plt
    plt.plot(t, sol[:, 0], 'r', label='x(t)')
    plt.plot(t, sol[:, 3], 'y', label='vx(t)')
    plt.plot(t, sol[:, 1], color='coral', label='y(t)')
    plt.plot(t, sol[:, 4], color='purple', label='vy(t)')
    plt.plot(t, sol[:, 2], 'b', label='z(t)')
    plt.plot(t, sol[:, 5], 'g', label='vz(t)')
    plt.legend(loc='best')
    plt.xlabel('t')
    plt.grid()
    plt.show()

def ballPlotGT(sol, t, y_gt, t_gt):
    '''
    y_gt: (1000, 6)
    '''
    import matplotlib.pyplot as plt
    ax1 = plt.subplot(1,3,1)
    ax1.set_title('z')
    plt.xlabel('t')
    plt.grid()
    plt.plot(t, sol[:, 2], 'b', label='z pred')
    plt.plot(t_gt, y_gt[:, 2], 'r', label='z groundtruth')
    plt.legend(loc='best')
    # plt.plot(t, sol[:, 5], 'g', label='vz(t)')
    # plt.plot(t_gt, y_gt[:, 5], 'g', label='vz(t)')
    

    ax2 = plt.subplot(1,3,2)
    ax2.set_title('x')
    plt.plot(t, sol[:, 0], 'b', label='x pred')
    plt.plot(t_gt, y_gt[:, 0], 'r', label='x groundtruth')
    plt.legend(loc='best')
    plt.xlabel('t')
    plt.grid()

    ax3 = plt.subplot(1,3,3)
    ax3.set_title('y')
    plt.plot(t, sol[:, 1], 'b', label='y pred')
    plt.plot(t_gt, y_gt[:, 1], 'r', label='y groundtruth')
    plt.legend(loc='best')
    plt.xlabel('t')
    plt.grid()

    plt.show()

def get_ball_traj(p0, v0, targetHeight: float = 1.0, dt: float = 0.02):
    """
    :param p00: 球初始位置
    :param v00: 球初始速度
    :param target_height: 给定的预测的下限高度
    :param ddt: 模拟步长
    :return: 球的位置轨迹(三次式)参数及速度轨迹(三次式)参数【共六条轨迹】
    """
    PosList, VelList = [], []

    pos, vel, g, simTime, CpuTime = np.array(p0), np.array(v0), np.array([0, 0, -9.81]), 0.0, time.time() # 初始状态

    PosList.append(pos)
    VelList.append(vel)
    # t = []
    # t.append(time)
    while True:
        if time.time() - CpuTime > 0.4:
            return None, None, None # 卡住0.1s退出
        if (pos[2] < targetHeight) and vel[2] < 0:
            break # 下落到指定高度退出
        simTime += dt
        acc = -0.05 * np.linalg.norm(vel) * vel + g -0.05 * vel -0.01 * np.sign(vel)
        pos += vel * dt + 0.5 * acc * dt ** 2
        vel += acc * dt
    
    return pos, vel, simTime

def get_ball_traj_ori(p00: np.ndarray, v00: np.ndarray, target_height: float = 1.0, kd_est: float = 0.05, ddt: float = 0.02):
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
    # t = []
    ball_pose.append(np.array(p00))
    ball_vel.append(np.array(v00))
    t_inl = 0.0
    # t.append(t_inl)
    # 根据公式迭代出球在空间的球姿，限制条件为球在指定的高度上方并且速度较小
    while (ball_pose[-1][2] >= target_height) and (abs(ball_vel[-1][2]) <= 20):
        p = ball_pose[-1] + ball_vel[-1]*ddt + 0.5*a*ddt**2
        # p = ball_pose[-1] + ball_vel[-1]*ddt
        ball_pose.append(p)
        v = ball_vel[-1] - kd_est * np.linalg.norm(ball_vel[-1])*ddt*ball_vel[-1] + a*ddt
        ball_vel.append(v)
        t_inl += ddt
        # t.append(t_inl)
    '''
    此时已大致完成了球的轨迹的求取，由于均为散点，因此多一步多项式拟合
    但也可以直接根据指定的碰撞高度，找到对应的相近的ball_pose及对应的时间，同样可以得到所需要的碰撞点的数据
    ————碰撞点坐标(x,y,z)、碰撞点球速(vx,vy,vz)
    '''
    target_ball_pose = ball_pose[-1]
    target_ball_vel = ball_vel[-1]
    return target_ball_pose,target_ball_vel,t_inl

if __name__ == '__main__':
    y0 = [0.0, 0.0, 1.0, 3.0, 4.0, 5.0]
    # y0 = [1.38, 0.22, 1.65, -0.04, -0.1, 2.35]

    startTime = time.time()
    x, v, t = ballPred(y0, output='x v t')
    print('x v t', time.time() - startTime, "seconds")
    print(x, v, t)

    startTime = time.time()
    [sol, t] = ballPred(y0, output='cutoff traj')
    print("cutoff traj", time.time() - startTime, "seconds")
    # ballPlot(sol, t)
    # ballPlotGT(sol, t, sol, t)

    startTime = time.time()
    [sol, t] = ballPred(y0, output='cutoff traj')
    print("cutoff traj", time.time() - startTime, "seconds")

    startTime = time.time()
    [xv, t] = ballPred(y0, output='xv t')
    print('xv t', time.time() - startTime, "seconds")
    print(xv, t, xv[0],xv[5])

    startTime = time.time()
    [xv, t] = ballPred(y0, output='xv t')
    print('xv t', time.time() - startTime, "seconds")
    print(xv, t, xv[0],xv[5])

    startTime = time.time()
    [x, v, t] = ballPred(y0, output='x v t')
    print('x v t', time.time() - startTime, "seconds")
    print(x, v, t)

    startTime = time.time()
    [x, v, t] = ballPred(y0, output='x v t')
    print('x v t', time.time() - startTime, "seconds")
    print(x, v, t)

    startTime = time.time()
    x, v, t = ballPred(y0, output='x v t')
    print('x v t', time.time() - startTime, "seconds")
    print(x, v, t)

    startTime = time.time()
    x, v, t = ballPred(y0, output='x v t')
    print(x, v, t, (time.time() - startTime) * 1000, "ms")

    startTime = time.time()
    x, v, t = get_ball_traj(y0[0:3], y0[3:6], targetHeight=1.0, dt=0.0002)
    print(x,v,t, (time.time() - startTime) * 1000, "ms")

    startTime = time.time()
    x, v, t = get_ball_traj(y0[0:3], y0[3:6], targetHeight=1.0, dt=0.002)
    print(x,v,t, (time.time() - startTime) * 1000, "ms")

    startTime = time.time()
    x, v, t = get_ball_traj(y0[0:3], y0[3:6], targetHeight=1.0, dt=0.02)
    print(x,v,t, (time.time() - startTime) * 1000, "ms")
