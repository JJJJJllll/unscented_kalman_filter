#!/usr/bin/env python3
import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped # 240724 jsl
import ballUtils_v1 as ballUtils # 240724 jsl
import signal # 用来退出程序
import sys # 用来退出程序
import time # 为了给算法计时
'''
I: kf_x3v3sss  O: 预测的落点速度时间
改了程序结构，预测落点应该放在callback里，主程序里不应该有while，应该spin
'''

def signal_handler(signal,frame):
    print('You pressed Ctrl + C!')
    sys.exit(0)

StartJuggle, StartJuggleTime = False, 0 # 颠球开始标志
BallVMax, BallVzMax, BigVel, BigVelCount = 0, 0, False, 0 # 速度达到最大、开始下降、且满足一定帧数
TargetHeight = 1.2 # 目标高度
BallNewMsg, ballXV = False, np.zeros(6) # 记录速度
lastPredInput = np.array([0.0, 0, 0, 0, 0, 0]) # 记录上次预测的结果

def natnet_callback(msg):
    global TargetHeight, StartJuggle
    if StartJuggle and abs(msg.pose.position.z - TargetHeight) < 0.1:
        rospy.loginfo("【natnet】当前位置[%5.2f, %5.2f, %5.2f]",msg.pose.position.x, msg.pose.position.y, msg.pose.position.z)

def ball_kf_callback(msg):
    global BallNewMsg, ballXV, StartJuggle, BigVel, BigVelCount, StartJuggleTime, BallVMax, BallVzMax, TargetHeight
    BallNewMsg = True
    ballXV[0], ballXV[1], ballXV[2] = msg.pose.position.x, msg.pose.position.y, msg.pose.position.z
    ballXV[3], ballXV[4], ballXV[5] = msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z
    ballvz = ballXV[5]
    ballv = np.linalg.norm([ballXV[3], ballXV[4], ballXV[5]])
    # 当球竖直速度突变时，等BigVelCount帧数据稳定后，发送颠球开始信号
    if ballvz > BallVzMax:
        BallVzMax = ballvz
        print("更新最大竖直速度%5.2f" % BallVzMax)
    if BallVzMax > 1 and ballvz < BallVzMax - 0.2:
        BigVel = True
    if BigVel and BigVelCount == 0:
        print("【球速拐点】达到最大、开始减小，最大竖直速度%5.2fm/s，当前竖直速度%5.2fm/s" % (BallVzMax, ballvz))
    if BigVel == True :#and ballv < BallVMax:
        BigVelCount += 1
    if BigVelCount == 1:
        StartJuggle, StartJuggleTime = True, time.time()
        print("【球抛出】当前位置[%5.2f,%5.2f,%5.2f]，当前速度[%5.2f,%5.2f,%5.2f]" % (ballXV[0], ballXV[1], ballXV[2],ballXV[3], ballXV[4], ballXV[5]), "当前时间", StartJuggleTime)
    
    # 实际落点
    if np.abs(ballXV[2] - TargetHeight) < 0.1:
        print("【球到落点】当前位置[%5.2f,%5.2f,%5.2f]，当前速度[%5.2f,%5.2f,%5.2f]，当前时间%5.2f，距抛出%5.2f" % (ballXV[0], ballXV[1], ballXV[2],ballXV[3], ballXV[4], ballXV[5], time.time(), time.time() - StartJuggleTime))

    # 预测落点
    global lastPredInput
    if StartJuggle and ballXV[2] > TargetHeight and np.linalg.norm(ballXV - lastPredInput) > 0.2:
        lastPredInput = np.array(ballXV) # 不每次都计算，而是有一点差距再算
        print("[Pred]输入[%5.2f,%5.2f,%5.2f,%5.2f,%5.2f,%5.2f]" % (ballXV[0], ballXV[1], ballXV[2], ballXV[3], ballXV[4],  ballXV[5]), "目标高度", TargetHeight)
        ball_predict_pose, ball_predict_vel, ball_predict_t = ballUtils.ballPred(ballXV, output='x v t', targetHeight=TargetHeight)
        print("[ode]落点[%5.2f,%5.2f,%5.2f]，速度[%5.2f,%5.2f,%5.2f]，离到落点还有%5.2fs，距抛出%5.2fms" % (ball_predict_pose[0], ball_predict_pose[1], ball_predict_pose[2], ball_predict_vel[0], ball_predict_vel[1], ball_predict_vel[2], ball_predict_t, (time.time() - StartJuggleTime) * 1000))

        x, v, t = ballUtils.get_ball_traj(ballXV[0:3], ballXV[3:6], targetHeight=TargetHeight, dt=0.0002)
        print("[0.0002]落点[%5.2f,%5.2f,%5.2f]，速度[%5.2f,%5.2f,%5.2f]，离到落点还有%5.2fs，距抛出%5.2fms" % (x[0], x[1], x[2], v[0], v[1], v[2], t, (time.time() - StartJuggleTime) * 1000))

        x, v, t = ballUtils.get_ball_traj(ballXV[0:3], ballXV[3:6], targetHeight=TargetHeight, dt=0.002)
        print("[0.002]落点[%5.2f,%5.2f,%5.2f]，速度[%5.2f,%5.2f,%5.2f]，离到落点还有%5.2fs，距抛出%5.2fms" % (x[0], x[1], x[2], v[0], v[1], v[2], t, (time.time() - StartJuggleTime) * 1000))

        x, v, t = ballUtils.get_ball_traj(ballXV[0:3], ballXV[3:6], targetHeight=TargetHeight, dt=0.02)
        print("[0.02]落点[%5.2f,%5.2f,%5.2f]，速度[%5.2f,%5.2f,%5.2f]，离到落点还有%5.2fs，距抛出%5.2fms" % (x[0], x[1], x[2], v[0], v[1], v[2], t, (time.time() - StartJuggleTime) * 1000))

    # else:
    #     print(StartJuggle, ballXV[5], "没进入预测", )

def main():
    rospy.init_node('jugglePred', anonymous=True)
    natnet_sub = rospy.Subscriber("/natnet_ros/ball/pose",
                                    PoseStamped,
                                    natnet_callback,
                                    queue_size = 2)
    ball_kf_sub = rospy.Subscriber("/kf_x3v3", # /kf_x3v3 /ukf_xv
                                PoseStamped,
                                ball_kf_callback,
                                queue_size = 2)

    # 预热ode一次
    print('下面这次是假预测、为了加速')
    preTime = time.time()
    a, b, c = ballUtils.ballPred([1,2,3,4,5,6], output='x v t', targetHeight=1)
    print("【假预测】落点[%5.2f, %5.2f, %5.2f]，速度[%5.2f, %5.2f, %5.2f]，离到落点还有%5.2fs，算法耗时%5.2fms" %( a[0], a[1], a[2], b[0], b[1], b[2], c, (time.time() - preTime)* 1000 ))
    # 循环
    rospy.spin()

    signal.signal(signal.SIGINT,signal_handler) # 随时退出程序
    print('Finish.')


if __name__ == "__main__":
    main()
