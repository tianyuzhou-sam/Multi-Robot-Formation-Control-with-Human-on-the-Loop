#!/usr/bin/env python3
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
from JackalSys import JackalSys
from OptimalControlProblem import OptimalControlProblem


class ModelPredictiveControl:
    configDict: dict  # a dictionary for parameters

    def __init__(self, configDict: dict, buildFlag=True):
        self.configDict = configDict

        # initialize JackalSys
        self.MyJackalSys = JackalSys(configDict, buildFlag)
        self.targets = [[5,5], [10,5], [10,10]]
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


    def run(self, iniState: np.array, iniInput: np.array, timeTotal: float):
        timeNow = 0.0  # [sec]
        stateNow = copy.deepcopy(iniState)
        inputNow = copy.deepcopy(iniInput)
        idx = 0

        # initialize trajectories for states, inputs, etc.
        timeTraj = np.array([timeNow], dtype=np.float64)
        # trajectory for states
        xTraj = np.zeros((1, self.MyJackalSys.dimStates), dtype=np.float64)
        xTraj[0, :] = stateNow
        # trajectory for input
        uTraj = np.zeros((1, self.MyJackalSys.dimInputs), dtype=np.float64)
        # trajectory for entire optimization time [sec]
        algTimeTraj = np.zeros(1, dtype=np.float64)
        # trajectory for Ipopt solution time [sec]
        ipoptTimeTraj = np.zeros(1, dtype=np.float64)
        # list for logging ipopt status
        logTimeTraj = list()
        logStrTraj = list()

        # target = self.targets[0]
        # self.MyOC = OptimalControlProblem(configDict, self.MyJackalSys, buildFlag, target)
        # load function to run optimization once
        if self.methodStr == "MPC":
            self.runOnce = lambda stateNow, timeNow: self._runOC(stateNow, timeNow)
        else:
            raise Exception("Wrong method in configDict!")

        reached = 0

        while (timeNow <= timeTotal-self.MyOC.dt) and (reached < self.numTargets):

            # solve
            ipoptTime, returnStatus, successFlag, algoTime, print_str, uNow = self.runOnce(stateNow, timeNow)

            # apply control and forward propagate dynamics
            xNext = self.MyJackalSys.discDynFun(stateNow, uNow)

            # update trajectory
            stateNow = np.array(xNext).reshape((1,-1)).flatten()
            xTraj = np.vstack((xTraj, stateNow))
            if idx <= 0.5:
                uTraj[idx, :] = uNow
                algTimeTraj[idx] = algoTime
                ipoptTimeTraj[idx] = ipoptTime
            else:
                uTraj = np.vstack((uTraj, uNow))
                algTimeTraj = np.append(algTimeTraj, algoTime)
                ipoptTimeTraj = np.append(ipoptTimeTraj, ipoptTime)

            if idx % 50 == 0:
                print(print_str)
            if not successFlag:
                if idx % 50 != 0:
                    print(print_str)
                print(returnStatus)
                logTimeTraj.append(timeNow)
                logStrTraj.append(returnStatus)

            # update time
            timeNow += self.MyOC.dt
            idx += 1
            timeTraj = np.append(timeTraj, timeNow)

            # if ((stateNow[0]-self.targets[reached][0])**2+(stateNow[1]-self.targets[reached][1])**2 <= self.offset**2):
            #     reached += 1
            #     if reached < self.numTargets:
            #         target = self.targets[reached]
            #         self.MyOC = OptimalControlProblem(configDict, self.MyJackalSys, buildFlag, target)

        print(print_str)
        result = {"timeTraj": timeTraj,
                  "xTraj": xTraj,
                  "uTraj": uTraj,
                  "ipoptTimeTraj": ipoptTimeTraj,
                  "logTimeTraj": logTimeTraj,
                  "logStrTraj": logStrTraj}

        return result

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

    buildFlag = True
    # buildFlag = False

    x0 = np.array([0, 0, 0])
    u0 = np.array([0, 0])
    T = 20

    # initialize MPC
    MyMPC = ModelPredictiveControl(configDict, buildFlag)
    result = MyMPC.run(x0, u0, T)
    print(result)
    MyMPC.visualize(result)
