#!/usr/bin/env python3
import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped # 240724 jsl
import ballUtils_v1 as ballUtils # 240724 jsl
import signal # 用来退出程序
import sys # 用来退出程序
import time # 为了给算法计时
'''
Sub: kf_x3v3  Pub + Print: 预测的落点速度时间
Feature:
1. 不用while改在callback里
2. 发送的HitTime是绝对时间kf_x3v3.header.stamp + ballHitTime, 要求kf_x3v3.header.stamp是绝对时间
3.(new) 配合test_latency可测通信延时
'''

def signal_handler(signal,frame):
    print('You pressed Ctrl + C!')
    sys.exit(0)

class jugglePred():
    StartJuggle, StartJuggleTime = False, 0 # 颠球开始标志
    BallVMax, BallVzMax, BigVel, BigVelCount = 0, 0, False, 0 # 速度达到最大、开始下降、且满足一定帧数
    TargetHeight = 0.8 # 目标高度
    lastPredInput = np.array([0.0, 0, 0, 0, 0, 0]) # 记录上次预测的结果
    pub_hit_pred = rospy.Publisher('hit_pred',
                                    PoseStamped,
                                    queue_size = 1)

    def __init__(self):
        natnet_sub = rospy.Subscriber("/natnet_ros/ball/pose",
                                    PoseStamped,
                                    self.natnet_callback,
                                    queue_size = 2)
        ball_kf_sub = rospy.Subscriber("/kf_x3v3", # /kf_x3v3 /ukf_xv
                                    PoseStamped,
                                    self.ball_kf_callback,
                                    queue_size = 2)
        test_latency_sub = rospy.Subscriber("/a",
                                    PoseStamped,
                                    self.test_latency_callback,
                                    queue_size = 2)
    
    def test_latency_callback(self, msg):
        SendTime = msg.header.stamp.to_time()
        RecvTime = time.time()
        print('RecvTime - SendTime%6.2fms, SendTime%10f, RecvTime%10f %10f' %(1000 * (RecvTime - SendTime), SendTime, RecvTime, rospy.Time.now().to_time()) )

    def natnet_callback(self, msg):
        if self.StartJuggle and abs(msg.pose.position.z - self.TargetHeight) < 0.1:
            rospy.loginfo("【natnet】当前位置[%5.2f, %5.2f, %5.2f]", msg.pose.position.x, msg.pose.position.y, msg.pose.position.z)

    def ball_kf_callback(self, msg):
        ballXV = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z, msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z])
        ballz, ballvz = msg.pose.position.z, msg.pose.orientation.z
        # 当球竖直速度突变时，等BigVelCount帧数据稳定后，发送颠球开始信号
        if ballvz > self.BallVzMax:
            self.BallVzMax = ballvz
            print("更新最大竖直速度%5.2f" % self.BallVzMax)
        if self.BallVzMax > 1 and ballvz < self.BallVzMax - 0.2:
            self.BigVel = True
        if self.BigVel and self.BigVelCount == 0:
            print("【球速拐点】达到最大、开始减小，最大竖直速度%5.2fm/s，当前竖直速度%5.2fm/s" % (self.BallVzMax, ballvz))
        if self.BigVel == True :#and ballv < BallVMax:
            self.BigVelCount += 1
        if self.BigVelCount == 1:
            self.StartJuggle, self.StartJuggleTime = True, time.time()
            print("【球抛出】当前位置[%5.2f,%5.2f,%5.2f]，当前速度[%5.2f,%5.2f,%5.2f]" % (ballXV[0], ballXV[1], ballXV[2],ballXV[3], ballXV[4], ballXV[5]), "当前时间", self.StartJuggleTime)
        
        # 实际落点
        if np.abs(ballz- self.TargetHeight) < 0.1 and ballvz < 0:
            print("【球到落点】当前位置[%5.2f,%5.2f,%5.2f]，当前速度[%5.2f,%5.2f,%5.2f]，当前时间%5.2f，距抛出%5.2f" % (ballXV[0], ballXV[1], ballXV[2],ballXV[3], ballXV[4], ballXV[5], time.time(), time.time() - self.StartJuggleTime))

        # 预测落点(不是每次都算，而是状态有一点变化再算)
        if self.StartJuggle and ballz > self.TargetHeight and np.linalg.norm(ballXV - self.lastPredInput) > 0.2:
            self.lastPredInput = ballXV 
            print("[Pred]输入[%5.2f,%5.2f,%5.2f,%5.2f,%5.2f,%5.2f]" % (ballXV[0], ballXV[1], ballXV[2], ballXV[3], ballXV[4],  ballXV[5]), "目标高度", self.TargetHeight)
            ballHitPos, ballHitVel, ballHitTime = ballUtils.ballPred(ballXV, output='x v t', targetHeight=self.TargetHeight)
            print("[ode]落点[%5.2f,%5.2f,%5.2f]，速度[%5.2f,%5.2f,%5.2f]，离到落点还有%5.2fs，距抛出%5.2fms" % (ballHitPos[0], ballHitPos[1], ballHitPos[2], ballHitVel[0], ballHitVel[1], ballHitVel[2], ballHitTime, (time.time() - self.StartJuggleTime) * 1000))

            # x, v, t = ballUtils.get_ball_traj(ballXV[0:3], ballXV[3:6], targetHeight=self.TargetHeight, dt=0.0002)
            # print("[0.0002]落点[%5.2f,%5.2f,%5.2f]，速度[%5.2f,%5.2f,%5.2f]，离到落点还有%5.2fs，距抛出%5.2fms" % (x[0], x[1], x[2], v[0], v[1], v[2], t, (time.time() - self.StartJuggleTime) * 1000))

            # x, v, t = ballUtils.get_ball_traj(ballXV[0:3], ballXV[3:6], targetHeight=self.TargetHeight, dt=0.002)
            # print("[0.002]落点[%5.2f,%5.2f,%5.2f]，速度[%5.2f,%5.2f,%5.2f]，离到落点还有%5.2fs，距抛出%5.2fms" % (x[0], x[1], x[2], v[0], v[1], v[2], t, (time.time() - self.StartJuggleTime) * 1000))

            # x, v, t = ballUtils.get_ball_traj(ballXV[0:3], ballXV[3:6], targetHeight=self.TargetHeight, dt=0.02)
            # print("[0.02]落点[%5.2f,%5.2f,%5.2f]，速度[%5.2f,%5.2f,%5.2f]，离到落点还有%5.2fs，距抛出%5.2fms" % (x[0], x[1], x[2], v[0], v[1], v[2], t, (time.time() - self.StartJuggleTime) * 1000))

            # 发布预测结果
            hit = PoseStamped()
            hit.header.stamp = rospy.Time.now()
            hit.pose.position.x = ballHitPos[0]
            hit.pose.position.y = ballHitPos[1]
            hit.pose.position.z = ballHitPos[2]
            hit.pose.orientation.x = ballHitVel[0]
            hit.pose.orientation.y = ballHitVel[1]
            hit.pose.orientation.z = ballHitVel[2]
            hit.pose.orientation.w = msg.header.stamp.to_time() + ballHitTime # 绝对落点时间，避免误差
            self.pub_hit_pred.publish(hit)
            print(hit.header.stamp, time.time(), hit.header.stamp.to_time())

        # else:
        #     print(StartJuggle, ballXV[5], "没进入预测", )

    def play(self):
        '''预热ode一次，然后就交给回调'''

        # 预热ode一次
        preTime = time.time()
        a, b, c = ballUtils.ballPred([1,2,3,4,5,6], output='x v t', targetHeight=1)
        print("[warm up ode]落点[%5.2f, %5.2f, %5.2f]，速度[%5.2f, %5.2f, %5.2f]，离到落点还有%5.2fs，算法耗时%5.2fms" %( a[0], a[1], a[2], b[0], b[1], b[2], c, (time.time() - preTime)* 1000 ))
        
        rospy.spin() # 循环

        signal.signal(signal.SIGINT,signal_handler) # 随时退出程序
        print('Finish.')
    
if __name__ == "__main__":
    rospy.init_node('jugglePred', anonymous=True)
    a = jugglePred()
    a.play()
