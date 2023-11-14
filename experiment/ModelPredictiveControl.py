#!/usr/bin/env python3
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.getcwd()+'/src')
from JackalSys import JackalSys
from OptimalControlProblem import OptimalControlProblem
import json
import transforms3d

import asyncio
import xml.etree.ElementTree as ET
import pkg_resources
import qtm

import math

import rospy
from geometry_msgs.msg import Twist

class ModelPredictiveControl:
    configDict: dict  # a dictionary for parameters

    def __init__(self, configDict: dict, buildFlag=True, config_file_name='config/config_jackal.json'):
        self.configDict = configDict

        # initialize JackalSys
        self.MyJackalSys = JackalSys(configDict, buildFlag)
        self.targets = [[0,0], [10,5], [10,10]]
        target = self.targets[0]
        # initialize OptimalControlProblem
        self.MyOC = OptimalControlProblem(configDict, self.MyJackalSys, buildFlag, target)
        
        # method
        try:
            self.methodStr = self.configDict["method"]
        # proposed MPC scheme by default
        except:
            print("No setting for method. Set MPC by default")
            self.methodStr = "MPC"

        
        self.numTargets = len(self.targets)
        self.offset = 0.1

        # Read the configuration from the json file
        self.json_file = open(config_file_name)
        self.config_data = json.load(self.json_file)
        self.IP_server = self.config_data["QUALISYS"]["IP_MOCAP_SERVER"]


        rospy.init_node('move_robot_node', anonymous=False)
        self.pub_move = rospy.Publisher("/cmd_vel", Twist, queue_size=100)

    def create_body_index(self, xml_string):
        """ Extract a name to index dictionary from 6-DOF settings xml """
        xml = ET.fromstring(xml_string)

        body_to_index = {}
        for index, body in enumerate(xml.findall("*/Body/Name")):
            body_to_index[body.text.strip()] = index

        return body_to_index

    def publish_vel(self):
        self.pub_move.publish(self.move)

    async def run(self, iniState: np.array, iniInput: np.array, timeTotal: float):
        # Connect to qtm
        connection = await qtm.connect(self.IP_server)

        # Connection failed?
        if connection is None:
            print("Failed to connect")
            return

        # Take control of qtm, context manager will automatically release control after scope end
        async with qtm.TakeControl(connection, "password"):
            pass

        # Get 6-DOF settings from QTM
        xml_string = await connection.get_parameters(parameters=["6d"])

        # parser for mocap rigid bodies indexing
        body_index = self.create_body_index(xml_string)
        # the one we want to access
        wanted_body = self.config_data["QUALISYS"]["NAME_SINGLE_BODY"]
        
        self.timeNow = 0.0  # [sec]
        self.stateNow = copy.deepcopy(iniState)
        self.inputNow = copy.deepcopy(iniInput)
        self.idx = 0

        # initialize trajectories for states, inputs, etc.
        self.timeTraj = np.array([self.timeNow], dtype=np.float64)
        # trajectory for states
        self.xTraj = np.zeros((1, self.MyJackalSys.dimStates), dtype=np.float64)
        self.xTraj[0, :] = self.stateNow
        # trajectory for input
        self.uTraj = np.zeros((1, self.MyJackalSys.dimInputs), dtype=np.float64)
        # trajectory for entire optimization time [sec]
        self.algTimeTraj = np.zeros(1, dtype=np.float64)
        # trajectory for Ipopt solution time [sec]
        self.ipoptTimeTraj = np.zeros(1, dtype=np.float64)
        # list for logging ipopt status
        self.logTimeTraj = list()
        self.logStrTraj = list()

        # target = self.targets[0]
        # self.MyOC = OptimalControlProblem(configDict, self.MyJackalSys, buildFlag, target)
        # load function to run optimization once
        if self.methodStr == "MPC":
            self.runOnce = lambda stateNow, timeNow: self._runOC(self.stateNow, self.timeNow)
        else:
            raise Exception("Wrong method in configDict!")

        self.reached = 0

        # while (timeNow <= timeTotal-self.MyOC.dt) and (reached < self.numTargets):

        #     # solve
        #     ipoptTime, returnStatus, successFlag, algoTime, print_str, uNow = self.runOnce(stateNow, timeNow)

        #     # apply control and forward propagate dynamics
        #     xNext = self.MyJackalSys.discDynFun(stateNow, uNow)

        #     # update trajectory
        #     stateNow = np.array(xNext).reshape((1,-1)).flatten()
        #     xTraj = np.vstack((xTraj, stateNow))
        #     if idx <= 0.5:
        #         uTraj[idx, :] = uNow
        #         algTimeTraj[idx] = algoTime
        #         ipoptTimeTraj[idx] = ipoptTime
        #     else:
        #         uTraj = np.vstack((uTraj, uNow))
        #         algTimeTraj = np.append(algTimeTraj, algoTime)
        #         ipoptTimeTraj = np.append(ipoptTimeTraj, ipoptTime)

        #     if idx % 50 == 0:
        #         print(print_str)
        #     if not successFlag:
        #         if idx % 50 != 0:
        #             print(print_str)
        #         print(returnStatus)
        #         logTimeTraj.append(timeNow)
        #         logStrTraj.append(returnStatus)

        #     # update time
        #     timeNow += self.MyOC.dt
        #     idx += 1
        #     timeTraj = np.append(timeTraj, timeNow)

        #     # if ((stateNow[0]-self.targets[reached][0])**2+(stateNow[1]-self.targets[reached][1])**2 <= self.offset**2):
        #     #     reached += 1
        #     #     if reached < self.numTargets:
        #     #         target = self.targets[reached]
        #     #         self.MyOC = OptimalControlProblem(configDict, self.MyJackalSys, buildFlag, target)

        # print(print_str)
        # result = {"timeTraj": timeTraj,
        #           "xTraj": xTraj,
        #           "uTraj": uTraj,
        #           "ipoptTimeTraj": ipoptTimeTraj,
        #           "logTimeTraj": logTimeTraj,
        #           "logStrTraj": logStrTraj}

        # return result

        def on_packet(packet):
            # Get the 6-DOF data
            bodies = packet.get_6d()[1]
            t_now = 0
            if (t_now < 10):
                # Extract one specific body
                t_now = time.time()
                wanted_index = body_index[wanted_body]
                position, rotation = bodies[wanted_index]
                # You can use position and rotation here. Notice that the unit for position is mm!
                # print(wanted_body)

                # rotation.matrix is a tuple with 9 elements.
                rotation_np = np.asarray(rotation.matrix, dtype=float).reshape(3, 3)

                # send 6-DOF data via TCP
                # concatenate the position and rotation matrix vertically
                # msg = np.asarray((position.x/1000.0, position.y/1000.0, position.z/1000.0) + rotation.matrix, dtype=float).tobytes()

                quat = transforms3d.quaternions.mat2quat(rotation_np)
                # print("quat")
                # print(quat)

                data = np.array([position.x/1000.0, position.y/1000.0, position.z/1000.0, quat[0], quat[1], quat[2], quat[3], t_now], dtype=float)

                # # for debugging
                # print("rotation matrix in array")
                # print(rotation.matrix)
                # print("rotation matrix in matrix")
                # print(rotation_np)

                roll_now, pitch_now, yaw_now = transforms3d.euler.quat2euler(quat, axes='sxyz')
                # print("yaw")
                print(math.degrees(yaw_now))
                # print("pitch")
                # print(math.degrees(pitch_now))
                # print("roll")
                # print(math.degrees(roll_now))

                # solve
                ipoptTime, returnStatus, successFlag, algoTime, print_str, uNow = self.runOnce(self.stateNow, self.timeNow)

                # apply control and forward propagate dynamics
                xNext = self.MyJackalSys.discDynFun(self.stateNow, uNow)
                xNext[0] = position[0]
                xNext[1] = position[1]

                # update trajectory
                self.stateNow = np.array(xNext).reshape((1,-1)).flatten()
                self.xTraj = np.vstack((self.xTraj, self.stateNow))
                if self.idx <= 0.5:
                    self.uTraj[self.idx, :] = uNow
                    self.algTimeTraj[self.idx] = algoTime
                    self.ipoptTimeTraj[self.idx] = ipoptTime
                else:
                    self.uTraj = np.vstack((self.uTraj, uNow))
                    self.algTimeTraj = np.append(self.algTimeTraj, algoTime)
                    self.ipoptTimeTraj = np.append(self.ipoptTimeTraj, ipoptTime)

                if self.idx % 50 == 0:
                    print(print_str)
                if not successFlag:
                    if idx % 50 != 0:
                        print(print_str)
                    print(returnStatus)
                    self.logTimeTraj.append(self.timeNow)
                    self.logStrTraj.append(returnStatus)

                # update time
                self.timeNow += self.MyOC.dt
                self.idx += 1
                self.timeTraj = np.append(self.timeTraj, self.timeNow)

                # if ((stateNow[0]-self.targets[reached][0])**2+(stateNow[1]-self.targets[reached][1])**2 <= self.offset**2):
                #     reached += 1
                #     if reached < self.numTargets:
                #         target = self.targets[reached]
                #         self.MyOC = OptimalControlProblem(configDict, self.MyJackalSys, buildFlag, target)

                print(print_str)
                result = {"timeTraj": self.timeTraj,
                            "xTraj": self.xTraj,
                            "uTraj": self.uTraj,
                            "ipoptTimeTraj": self.ipoptTimeTraj,
                            "logTimeTraj": self.logTimeTraj,
                            "logStrTraj": self.logStrTraj}
                # print(result)

                move =Twist()
                move.linear.x = uNow[0]
                move.angular.z = uNow[1]
                self.pub_move.publish(move)   
            
            else:
                # error
                raise Exception("There is no such a rigid body!")

        # Start streaming frames
        # Make sure the component matches with the data fetch function, for example: packet.get_6d() with "6d"
        # Reference: https://qualisys.github.io/qualisys_python_sdk/index.html
        await connection.stream_frames(components=["6d"], on_packet=on_packet)

    def _runOC(self, stateNow, timeNow):
        t0 = time.time()
        xTrajNow, uTrajNow, timeTrajNow, ipoptTime, returnStatus, successFlag = self.MyOC.solve(stateNow, timeNow)
        t1 = time.time()
        algoTime = t1 - t0
        print_str = "Sim time [sec]: " + str(round(timeNow, 1)) + "   Comp. time [sec]: " + str(round(algoTime, 3))
        # apply control
        uNow = uTrajNow[0, :]
        return ipoptTime, returnStatus, successFlag, algoTime, print_str, uNow


    def visualize(self, result: dict, matFlag=False, legendFlag=True, titleFlag=True, blockFlag=True):
        """
        If result is loaded from a .mat file, matFlag = True
        If result is from a seld-defined dict variable, matFlag = False
        """
        timeTraj = result["timeTraj"]
        xTraj = result["xTraj"]
        uTraj = result["uTraj"]

        if matFlag:
            timeTraj = timeTraj[0, 0].flatten()
            xTraj = xTraj[0, 0]
            uTraj = uTraj[0, 0]

        # trajectories for states
        fig1, ax1 = plt.subplots(2, 1)
        if titleFlag:
            fig1.suptitle("State Trajectory")
        ax1[0].plot(xTraj[:,0], xTraj[:,1], color="blue", linewidth=2)

        ax1[0].set_xlabel("x [m]")
        ax1[0].set_ylabel("y [m]")

        ax1[1].plot(timeTraj, xTraj[:,0], color="blue", linewidth=2)
        ax1[1].plot(timeTraj, xTraj[:,1], color="red", linewidth=2)
        ax1[1].set_xlabel("time [sec]")
        ax1[1].set_ylabel("x,y [m]")

        # trajectories for inputs
        fig2, ax2 = plt.subplots(2, 1)
        if titleFlag:
            fig2.suptitle("Input Trajectory")
        # trajectory for current
        ax2[0].plot(timeTraj[:-1], uTraj[:,0], color="blue", linewidth=2)
        # for input bounds
        ax2[0].plot(timeTraj[:-1],
            self.MyOC.lin_vel_lb*np.ones(timeTraj[:-1].size),
            color="black", linewidth=2, linestyle="dashed")
        ax2[0].plot(timeTraj[:-1],
            self.MyOC.lin_vel_ub*np.ones(timeTraj[:-1].size),
            color="black", linewidth=2, linestyle="dashed")
        # ax2[0].set_xlabel("time [sec]")
        ax2[0].set_ylabel(r'$v \  [\mathrm{m/s}]$')

        ax2[1].plot(timeTraj[:-1], uTraj[:,1], color="blue", linewidth=2)
        # for input bounds
        ax2[1].plot(timeTraj[:-1],
            self.MyOC.ang_vel_lb*np.ones(timeTraj[:-1].size),
            color="black", linewidth=2, linestyle="dashed")
        ax2[1].plot(timeTraj[:-1],
            self.MyOC.ang_vel_ub*np.ones(timeTraj[:-1].size),
            color="black", linewidth=2, linestyle="dashed")
        ax2[1].set_xlabel("time [sec]")
        ax2[1].set_ylabel(r'$w\  [\mathrm{rad/s}]$')
        plt.tight_layout()
        plt.show(block=blockFlag)


if __name__ == '__main__':
    # dictionary for configuration
    # dt for Euler integration
    configDict = {"dt": 0.1, "stepNumHorizon": 10, "startPointMethod": "zeroInput"}
    config_file_name = 'config/config_jackal.json'

    buildFlag = True
    # buildFlag = False

    

    x0 = np.array([0, 0, 0])
    u0 = np.array([0, 0])
    T = 20

    # initialize MPC
    MyMPC = ModelPredictiveControl(configDict, buildFlag, config_file_name)

    # Run our asynchronous main function forever
    asyncio.ensure_future(MyMPC.run(x0, u0, T))
    asyncio.get_event_loop().run_forever()

