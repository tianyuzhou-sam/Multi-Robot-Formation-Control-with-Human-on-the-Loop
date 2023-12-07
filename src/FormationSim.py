#!/usr/bin/env python3
from random import random
import numpy as np
from numpy.linalg import inv
from numpy import linalg as LA
from scipy import integrate
import scipy
from random import random
import matplotlib.pyplot as plt
import csv
import os
import sys
import time
from JackalSys import JackalSys
from OptimalControlJackal import OptimalControlJackal
from QuadSys import QuadSys
from plot_traj import Simulator

class FormationSim:
    configDict: dict  # a dictionary for parameters
    dimStates: int  # dimension of states
    dimInputs: int  # dimension of inputs

    def __init__(self, configDict: dict, targets: [0,0], iniJackal: [0,0,0], iniQuad: [0,0,0], desire: [[0,0,0]], t_change: np.inf, buildFlag=True, saveFlag=False):
        self.saveFlag = saveFlag
        self.num_agents = len(iniQuad)
        self.stateTransform = np.array([[1,0,0,0,0,0,0,0,0,0,0,0],
                                        [0,1,0,0,0,0,0,0,0,0,0,0],
                                        [0,0,0,1,0,0,0,0,0,0,0,0],
                                        [0,0,0,0,1,0,0,0,0,0,0,0]])
        self.configDict = configDict
        self.dt = configDict['dt']
        self.MyJackalSys = JackalSys(configDict, buildFlag)
        self.Jackal = OptimalControlJackal(configDict, self.MyJackalSys, buildFlag)
        self.maxTime = 0

        self.noiseAmp = 0.1

        self.targets = targets
        self.reached = 0
        self.numTargets = len(targets)
        self.epsilon = 0.01
        self.hold = 0

        self.Quad = QuadSys(configDict)

        self.g = self.Quad.g
        self.m = self.Quad.m
        self.Ix = self.Quad.Ix
        self.Iy = self.Quad.Iy
        self.Iz = self.Quad.Iz

        self.x_Jackal = iniJackal

        self.x_Quad = np.zeros((self.num_agents, self.Quad.dimStates))
        for idx in range(self.num_agents):
            self.x_Quad[idx][0] = iniQuad[idx][0]
            self.x_Quad[idx][1] = iniQuad[idx][1]
            self.x_Quad[idx][2] = iniQuad[idx][2]

        self.t_change = t_change
        self.nextChange = self.t_change
        self.formation_idx = 0
        self.desire = desire
        self.z = np.zeros((self.num_agents, self.Quad.dimStates))
        if t_change is np.inf:
            for idx in range(self.num_agents):
                self.z[idx][0] = desire[idx][0]
                self.z[idx][1] = desire[idx][1]
                self.z[idx][2] = desire[idx][2]
        else:
            for idx in range(self.num_agents):
                self.nextChange = self.t_change[0]
                self.z[idx][0] = desire[self.formation_idx][idx][0]
                self.z[idx][1] = desire[self.formation_idx][idx][1]
                self.z[idx][2] = desire[self.formation_idx][idx][2]


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

        # self.MySimulator = Simulator

    def run(self):
        u = np.zeros((self.Quad.dimInputs, 1))

        # ax = self.MySimulator.create_realtime_plot(realtime_flag=True, cluster_legend_flag=True, path_legend_flag=True)

        timeNow = 0.0
        timeTraj = np.array([timeNow], dtype=np.float64)
        target = self.targets[0]
        # v_save = np.zeros((6,1))

        # X = np.transpose(X)[-1]
        quad_state_list = timeNow
        for idx in range(self.num_agents):
            quad_state_list = np.hstack((quad_state_list, self.x_Quad[idx][0:3], 0, 0, 0, 0, 0, 0, 0, 0, 0))
        leader_state_list = np.hstack((self.x_Jackal[0], self.x_Jackal[1],0,0))

        self.runOnce = lambda stateNow, timeNow, target: self._runOC(self.x_Jackal, timeNow, target)
        while (self.reached < self.numTargets) or timeNow <= self.maxTime:    

            # solve
            if timeNow > self.hold:
                ipoptTime, returnStatus, successFlag, algoTime, print_str, uNow = self.runOnce(self.x_Jackal, timeNow, target)
            else:
                uNow = [0,0]

            JackalObserved = [self.x_Jackal[0], self.x_Jackal[1], 0, uNow[0]*np.cos(self.x_Jackal[2]), uNow[0]*np.sin(self.x_Jackal[2]), 0, 0, 0, 0, 0, 0, 0]
            # JackalObserved = [self.x_Jackal[0] + self.noiseAmp*(random()-0.5), self.x_Jackal[1] + self.noiseAmp*(random()-0.5), 0, uNow[0]*np.cos(self.x_Jackal[2]), uNow[0]*np.sin(self.x_Jackal[2]), 0, 0, 0, 0, 0, 0, 0]
            
            # apply control and forward propagate dynamics
            JackalStateNext = self.MyJackalSys.discDynFun(self.x_Jackal, uNow)
            # update trajectory
            self.x_Jackal = np.array(JackalStateNext).reshape((1,-1)).flatten()
            leader_state_list = np.vstack((leader_state_list, np.array([self.x_Jackal[0], self.x_Jackal[1], 0, 0])))

            # update time
            timeNow += self.dt

            if (self.reached < self.numTargets):
                if ((self.x_Jackal[0] - self.targets[self.reached][0])**2 + (self.x_Jackal[1] - self.targets[self.reached][1])**2 <= self.epsilon**2):
                    self.reached += 1
                    if self.reached < self.numTargets:
                        target = self.targets[self.reached]


            quad_state = np.zeros((self.num_agents*self.Quad.dimStates+1))
            quad_state[0] = timeNow   
            for idx in range(self.num_agents):
                u = -np.matmul(self.K, self.x_Quad[idx] - self.z[idx] - JackalObserved)
                u = u.tolist()[0]

                u[0] = u[0] + self.m*self.g

                newX = self.Quad._discDynFun(self.x_Quad[idx], u)
                for jdx in range(self.Quad.dimStates):
                    self.x_Quad[idx][jdx] = newX[jdx]
                    quad_state[idx*self.Quad.dimStates+jdx+1] = newX[jdx]
                

            quad_state_list = np.vstack((quad_state_list, quad_state))

            if timeNow >= self.nextChange:
                self.formation_idx += 1
                self.nextChange = self.t_change[self.formation_idx]
                for idx in range(self.num_agents):
                    self.z[idx][0] = self.desire[self.formation_idx][idx][0]
                    self.z[idx][1] = self.desire[self.formation_idx][idx][1]
                    self.z[idx][2] = self.desire[self.formation_idx][idx][2]

            # fig1, ax1 = plt.subplots()
            # ax1.plot(self.x_Jackal[0], self.x_Jackal[1], marker = 'o')
            # for idx in range(self.num_agents):
            #     ax1.plot(self.x_Quad[idx][0], self.x_Quad[idx][1], marker = '^')
            # for idx in range(self.numTargets):
            #     ax1.plot(self.targets[idx][0], self.targets[idx][1], marker = '*')
            # ax1.set_xlabel('x (m)')
            # ax1.set_ylabel('y (m)')
            # ax1.set_title('Formation')
            # ax1.set_xlim([-2.5, 2.5])
            # ax1.set_ylim([-2.5, 2.5])
            # plt.show()

        if self.saveFlag:
            for idx in range(self.num_agents):
                array_csv = quad_state_list[:,0]
                array_csv = np.vstack((array_csv, quad_state_list[:,idx*self.Quad.dimStates+1]))
                array_csv = np.vstack((array_csv, quad_state_list[:,idx*self.Quad.dimStates+2]))
                array_csv = np.vstack((array_csv, quad_state_list[:,idx*self.Quad.dimStates+3]))
                filename_csv = os.path.expanduser("~") + "/github/Multi-agent-Formation-Control-With-Human-Guidance/mambo_0" + str(idx+1) + ".csv"
                np.savetxt(filename_csv, array_csv, delimiter=",")
            
            array_csv = quad_state_list[:,0]
            array_csv = np.vstack((array_csv, leader_state_list[:,0]))
            array_csv = np.vstack((array_csv, leader_state_list[:,1]))
            filename_csv = os.path.expanduser("~") + "/github/Multi-agent-Formation-Control-With-Human-Guidance/jackal.csv"
            np.savetxt(filename_csv, array_csv, delimiter=",")

            


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
