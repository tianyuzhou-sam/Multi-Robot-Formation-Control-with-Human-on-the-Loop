#!/usr/bin/env python3
from random import random
import numpy as np
from numpy.linalg import inv
from numpy import linalg as LA
from scipy import integrate
import scipy
import matplotlib.pyplot as plt
import csv
import os
import sys
import time
from JackalSys import JackalSys
from OptimalControlJackal import OptimalControlJackal
from QuadSys import QuadSys

class FormationSim:
    configDict: dict  # a dictionary for parameters
    dimStates: int  # dimension of states
    dimInputs: int  # dimension of inputs

    def __init__(self, configDict: dict, targets: [0,0], iniJackal: [0,0,0], iniQuad: [0,0,0], desire: [0,0,0], buildFlag=True, saveFlag=False):
        self.num_agents = len(iniQuad)
        self.stateTransform = np.array([[1,0,0,0,0,0,0,0,0,0,0,0],
                                        [0,1,0,0,0,0,0,0,0,0,0,0],
                                        [0,0,0,1,0,0,0,0,0,0,0,0],
                                        [0,0,0,0,1,0,0,0,0,0,0,0]])
        self.configDict = configDict
        self.dt = configDict['dt']
        self.MyJackalSys = JackalSys(configDict, buildFlag)
        self.Jackal = OptimalControlJackal(configDict, self.MyJackalSys, buildFlag)

        self.targets = targets
        self.reached = 0
        self.numTargets = len(targets)
        self.offset = 0.1
        self.hold = 2

        self.g = 9.81
        self.m = 0.2
        self.Ix = 8.1 * 1e-3
        self.Iy = 8.1 * 1e-3
        self.Iz = 14.2 * 1e-3

        self.Quad = QuadSys(configDict, self.g, self.m, self.Ix, self.Iy, self.Iz)

        self.x_Jackal = iniJackal

        self.x_Quad = np.zeros((self.num_agents, self.Quad.dimStates))
        for idx in range(self.num_agents):
            self.x_Quad[idx][0] = iniQuad[idx][0]
            self.x_Quad[idx][1] = iniQuad[idx][1]
            self.x_Quad[idx][2] = iniQuad[idx][2]

        self.z = np.zeros((self.num_agents, self.Quad.dimStates))
        for idx in range(self.num_agents):
            self.z[idx][0] = desire[idx][0]
            self.z[idx][1] = desire[idx][1]
            self.z[idx][2] = desire[idx][2]


        self.A = np.array([[0,0,0,1,0,0,0,0,0,0,0,0],
                           [0,0,0,0,1,0,0,0,0,0,0,0],
                           [0,0,0,0,0,1,0,0,0,0,0,0],
                           [0,0,0,0,0,0,0,self.g,0,0,0,0],
                           [0,0,0,0,0,0,-self.g,0,0,0,0,0],
                           [0,0,0,0,0,0,0,0,0,0,0,0],
                           [0,0,0,0,0,0,0,0,0,1,0,0],
                           [0,0,0,0,0,0,0,0,0,0,1,0],
                           [0,0,0,0,0,0,0,0,0,0,0,1],
                           [0,0,0,0,0,0,0,0,0,0,0,0],
                           [0,0,0,0,0,0,0,0,0,0,0,0],
                           [0,0,0,0,0,0,0,0,0,0,0,0]])

        self.B = np.array([[0,0,0,0],
                           [0,0,0,0],
                           [0,0,0,0],
                           [0,0,0,0],
                           [0,0,0,0],
                           [1/self.m,0,0,0],
                           [0,0,0,0],
                           [0,0,0,0],
                           [0,0,0,0],
                           [0,1/self.Ix,0,0],
                           [0,0,1/self.Iy,0],
                           [0,0,0,1/self.Iz]])

        self.Ad = np.eye(self.Quad.dimStates) + self.dt*self.A
        self.Bd = self.dt*self.B

        self.Q = np.eye(self.Quad.dimStates)*10
        self.R = np.eye(self.Quad.dimInputs)*1
        
        
        P = np.matrix(scipy.linalg.solve_discrete_are(self.Ad, self.Bd, self.Q, self.R))
        self.K = np.matrix(np.matmul(scipy.linalg.inv(self.R + self.Bd.T*P*self.Bd),self.Bd.T)*P*self.Ad)



    def run(self):


        u = np.zeros((self.Quad.dimInputs, 1))

        # X_save = np.transpose(x0)

        # X = x0
        # X = np.vstack((X, np.transpose(np.kron(np.transpose(x0), np.transpose(x0)))))
        # X = np.vstack((X, np.kron(x0, np.zeros((un, 1)))))

        timeNow = 0.0
        timeTraj = np.array([timeNow], dtype=np.float64)
        target = self.targets[0]
        # v_save = np.zeros((6,1))

        # X = np.transpose(X)[-1]


        self.runOnce = lambda stateNow, timeNow, target: self._runOC(self.x_Jackal, timeNow, target)
        # x = np.reshape(X_save[-1], (xn,1))
        while (self.reached < self.numTargets):    

            # solve
            if timeNow > self.hold:
                ipoptTime, returnStatus, successFlag, algoTime, print_str, uNow = self.runOnce(self.x_Jackal, timeNow, target)
            else:
                uNow = [0,0]

            JackalObserved = [self.x_Jackal[0], self.x_Jackal[1], 0, uNow[0]*np.cos(self.x_Jackal[2]), uNow[0]*np.sin(self.x_Jackal[2]), 0, 0, 0, 0, 0, 0, 0]
            
            # apply control and forward propagate dynamics
            JackalStateNext = self.MyJackalSys.discDynFun(self.x_Jackal, uNow)
            # update trajectory
            self.x_Jackal = np.array(JackalStateNext).reshape((1,-1)).flatten()


            # update time
            timeNow += self.dt
            timeTraj = np.append(timeTraj, timeNow)

            if (self.reached < self.numTargets):
                if ((self.x_Jackal[0] - self.targets[self.reached][0])**2 + (self.x_Jackal[1] - self.targets[self.reached][1])**2 <= self.offset**2):
                    self.reached += 1
                    if self.reached < self.numTargets:
                        target = self.targets[self.reached]

            for idx in range(self.num_agents):
                u = -np.matmul(self.K, self.x_Quad[idx] - self.z[idx] - JackalObserved)
                u = u.tolist()[0]

                # new_v = np.vstack((dx[3:6], dx[9:12]))

                # v_save = np.hstack((v_save, new_v))

                u[0] = u[0] + self.m*self.g

                newX = self.Quad._discDynFun(self.x_Quad[idx], u)
                for jdx in range(self.Quad.dimStates):
                    self.x_Quad[idx][jdx] = newX[jdx]

                # t_save = np.hstack((t_save, t_save[-1]+dt))
                # X_save = np.vstack((X_save, np.transpose(x)))

            fig1, ax1 = plt.subplots()
            ax1.plot(self.x_Jackal[0], self.x_Jackal[1], marker = 'o')
            for idx in range(self.num_agents):
                ax1.plot(self.x_Quad[idx][0], self.x_Quad[idx][1], marker = '^')
            for idx in range(self.numTargets):
                ax1.plot(self.targets[idx][0], self.targets[idx][1], marker = '*')
            ax1.set_xlabel('x (m)')
            ax1.set_ylabel('y (m)')
            ax1.set_xlim([-2.5, 2.5])
            ax1.set_ylim([-2.5, 2.5])
            plt.show()


        # fig1, ax1 = plt.subplots(3,1)
        # ax1[0].plot(t_save, X_save[:,0])
        # ax1[0].set_xlabel('t (s)')
        # ax1[0].set_ylabel('x (m)')
        # ax1[1].plot(t_save, X_save[:,1])
        # ax1[1].set_xlabel('t (s)')
        # ax1[1].set_ylabel('y (m)')
        # ax1[2].plot(t_save, X_save[:,2])
        # ax1[2].set_xlabel('t (s)')
        # ax1[2].set_ylabel('z (m)')

        # fig2, ax2 = plt.subplots(3,1)
        # ax2[0].plot(t_save, X_save[:,7])
        # ax2[0].set_xlabel('t (s)')
        # ax2[0].set_ylabel('roll (rad))')
        # ax2[1].plot(t_save, X_save[:,8])
        # ax2[1].set_xlabel('t (s)')
        # ax2[1].set_ylabel('pitch (rad)')
        # ax2[2].plot(t_save, X_save[:,9])
        # ax2[2].set_xlabel('t (s)')
        # ax2[2].set_ylabel('yaw (rad)')


        # plt.show()
    def _runOC(self, stateNow, timeNow, target):
        t0 = time.time()
        xTrajNow, uTrajNow, timeTrajNow, ipoptTime, returnStatus, successFlag = self.Jackal.solve(stateNow, timeNow, target)
        t1 = time.time()
        algoTime = t1 - t0
        print_str = "Sim time [sec]: " + str(round(timeNow, 1)) + "   Comp. time [sec]: " + str(round(algoTime, 3))
        # apply control
        uNow = uTrajNow[0, :]
        return ipoptTime, returnStatus, successFlag, algoTime, print_str, uNow

if __name__ == '__main__':
    # dictionary for configuration
    # dt for Euler integration
    configDict = {"dt": 0.1, "stepNumHorizon": 10, "startPointMethod": "zeroInput"}

    buildFlag = True
    saveFlag = False

    targets = [[1,1],[2,1]]
    iniJackal = [0,0,0]
    iniQuad = [[0,0,0],[0,1,0],[0,-1,0]]
    desire = [[-0.25,0.5,1],[-0.25,-0.5,1],[0.25,0,1]]

    MySim = FormationSim(configDict, targets, iniJackal, iniQuad, desire, buildFlag, saveFlag)
    MySim.run()
