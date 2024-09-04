import numpy as np
import time

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

    print(sol.shape)
    # print(sol[:,2][::-1])
    print(sol[:,2][::-1].max())
    print(targetHeight)
    # print(np.where(sol[:,2][::-1] > targetHeight))
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

if __name__ == '__main__':
    y0 = [0, 0, 0, 3, 4, 5]
    # y0 = [1.38, 0.22, 1.65, -0.04, -0.1, 2.35]

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
    [x, v, t] = ballPred(y0, output='x v t')
    print('x v t', time.time() - startTime, "seconds")
    print(x, v, t)

    startTime = time.time()
    x, v, t = ballPred(y0, output='x v t')
    print('x v t', time.time() - startTime, "seconds")
    print(x, v, t)

    startTime = time.time()
    x, v, t = ballPred([1,2,3,4,5,6], output='x v t', targetHeight=1.2)
    print('x v t', time.time() - startTime, "seconds")
    print(x, v, t)

    startTime = time.time()
    print("特殊的测试用例")
    x, v, t = ballPred([2.28979809, 1.41803734, 0.65326352, -3.02483778, -1.80875359, 3.35874392], output='x v t', targetHeight=1.2)
    print('x v t', time.time() - startTime, "seconds")
    print(x, v, t)