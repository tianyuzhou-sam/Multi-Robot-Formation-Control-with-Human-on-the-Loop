#!/usr/bin/env python3
from asyncore import read
import imp
import os
import sys
import asyncio
import time
import json
from itertools import chain
import matplotlib.pyplot as plt
import numpy as np
# from pandas import array
import scipy
from scipy import integrate
from random import random
import csv
from sympy import quadratic_residues, true
sys.path.append('/home/lab-user/tianyu/Mambo-Tracking-Interface/lib')
import csv_helper
from UdpProtocol import UdpProtocol
sys.path.append(os.getcwd()+'/src')
from QuadSys import QuadSys

class FormationPlanner:
    num_agents: int  # number of agents
    planning_frequency: float  # time step
    list_AgentFSMExp: list  # a list of AgentFSMExp objects
    num_cluster: int  # number of clusters for task decomposition, mission planning
    number_of_iterations: int  # number of iterations for task decomposition, mission planning
    config_data_list: list  # a list of dictionaries for agents configurations
    time_name: str
    height_fly: float
    distance: np.array
    iter_max: int
    epsilon : float
    xn_single: int
    xn: int
    un: int

    def __init__(self):
        """
        Initialize a PlannerMocap Object. This is used with Motion Capture System.
        """
        self.num_agents = 3  # number of agents

        # load multiple agents' configuration as a list of dictionaries
        self.config_data_list = list()
        for idx in range(self.num_agents):
            file_name = os.getcwd() + "/experiment/config/config_aimslab_" + str(idx+1) + ".json"
            self.config_data_list.append(json.load(open(file_name)))

        self.mocap_type = "QUALISYS"
        # the address for publishing agents positions from mocap
        self.state_address_list = list()
        for idx in range(self.num_agents):
            self.state_address_list.append(
                (self.config_data_list[idx][self.mocap_type]["IP_STATES_ESTIMATION"],
                int(self.config_data_list[idx][self.mocap_type]["PORT_POSITION_PLANNER"])))

        # the address for publishing target positions from mocap
        self.server_address_target = (self.config_data_list[0][self.mocap_type]["IP_OBS_POSITION"],
                                   int(self.config_data_list[0][self.mocap_type]["PORT_OBS_POSITION"]))

    async def run_planner(self, formation):
        """
        Run the planner online with Motion Capture System.
        """
        # self.list_AgentFSMExp = list()  # a list for Agent Finite State Machine
        self.dt = 0.5
        # number of clusters for task decomposition
        self.num_cluster = self.num_agents
        # number of iterations for task decomposition
        self.number_of_iterations = 500
        self.iter_max = 10
        self.epsilon = 10e-3

        # flying time
        fly_time = 100


        self.g = 9.81
        self.m = 0.2
        self.Ix = 8.1 * 1e-3
        self.Iy = 8.1 * 1e-3
        self.Iz = 14.2 * 1e-3
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
        

        configDict = {"dt": self.dt, "stepNumHorizon": 10, "startPointMethod": "zeroInput"}
        self.MyQuad = QuadSys(configDict, self.g, self.m, self.Ix, self.Iy, self.Iz)

        
        # dimension of group matrices
        self.dim_xn = len(self.B)                          # single agent
        self.dim_un = len(self.B[0])                       # number of columns

        self.z = np.zeros((self.num_agents, self.dim_xn))
        for idx in range(self.num_agents):
            self.z[idx][0] = formation[idx][0]
            self.z[idx][1] = formation[idx][1]
            self.z[idx][2] = formation[idx][2]
        
        self.Ad = np.eye(self.dim_xn) + self.dt*self.A
        self.Bd = self.dt*self.B

        self.Q = np.eye(self.dim_xn)*10
        self.R = np.eye(self.dim_un)*1
        
        
        P = np.matrix(scipy.linalg.solve_discrete_are(self.Ad, self.Bd, self.Q, self.R))
        self.K = np.matrix(np.matmul(scipy.linalg.inv(self.R + self.Bd.T*P*self.Bd),self.Bd.T)*P*self.Ad)


        # remove all the existing files in the trajectory directory
        for idx in range(self.num_agents):
            directory_delete = os.path.expanduser("~") + "/tianyu/Mambo-Tracking-Interface" + \
                               self.config_data_list[idx]["DIRECTORY_TRAJ"] + "*"
            csv_helper.remove_traj_ref_lib(directory_delete)
        directory_delete = os.path.expanduser("~") + "/tianyu/Mambo-Tracking-Interface/scripts_aimslab/traj_csv_files/jackal/" + "*"
        csv_helper.remove_traj_ref_lib(directory_delete)

        # create a customized UDP protocol for subscribing states from mocap
        loop_states_01 = asyncio.get_running_loop()
        _, protocol_states_01 = await loop_states_01.create_datagram_endpoint(
            UdpProtocol, local_addr=self.state_address_list[0], remote_addr=None)

        loop_states_02 = asyncio.get_running_loop()
        _, protocol_states_02 = await loop_states_02.create_datagram_endpoint(
            UdpProtocol, local_addr=self.state_address_list[1], remote_addr=None)

        loop_states_03 = asyncio.get_running_loop()
        _, protocol_states_03 = await loop_states_03.create_datagram_endpoint(
            UdpProtocol, local_addr=self.state_address_list[2], remote_addr=None)

        # self.transport_states_list = [transport_states_01, transport_states_02, transport_states_03]
        self.protocol_states_list = [protocol_states_01, protocol_states_02, protocol_states_03]
        
        # create a customized UDP protocol for subscribing target positions from mocap
        loop_target = asyncio.get_running_loop()
        _, self.protocol_loop_target = await loop_target.create_datagram_endpoint(
            UdpProtocol, local_addr=self.server_address_target, remote_addr=None)

        # get the agent home position
        agents_position_list = await self.update_states_mocap()
        
        # get target position
        target_position = await self.update_target_mocap()

        self.time_name = time.strftime("%Y%m%d%H%M%S")

        time_begin = time.time()
        time_used = 0  # initialize the global time as 0
        quad_state_list = time.time() - time_begin
        for idx in range(self.num_agents):
            quad_state_list = np.hstack((quad_state_list, agents_position_list[idx][0:3], 0, 0, 0, 0, 0, 0, 0, 0, 0))
        target_state_list = np.hstack((target_position[0], target_position[1],0,0))

        time_start_fly = time.time()
        while(time_used < fly_time):
            t_start = time.time()

            # update the agent position
            agents_position_list = await self.update_states_mocap()

            # update target position
            target_position = await self.update_target_mocap()

            # do formation
            quad_state_list, target_state_list = self.formation(time_begin, agents_position_list, target_state_list, target_position, quad_state_list)

            # plt.pause(1E-9)
            time_sleep = max(0, self.dt - time.time() + t_start)
            time_used = time.time() - time_start_fly
            print("Current Time [sec]: " + str(time.time()-time_begin))
            await asyncio.sleep(time_sleep)


    def formation(self, time_begin, agents_position_list, target_state_list, target_position, quad_state_list):

        old_v = np.zeros((3*self.num_agents))
        old_target_v = np.zeros((2))

        for idx in range(self.num_agents):
            if quad_state_list.size == self.num_agents*self.dim_xn+1:
                old_v[idx*3] = quad_state_list[idx*self.dim_xn+4]
                old_v[idx*3+1] = quad_state_list[idx*self.dim_xn+5]
                old_v[idx*3+2] = quad_state_list[idx*self.dim_xn+6]
                old_target_v[0] = target_state_list[2]
                old_target_v[1] = target_state_list[3]
            else:
                old_v[idx*3] = quad_state_list[-1][idx*self.dim_xn+4]
                old_v[idx*3+1] = quad_state_list[-1][idx*self.dim_xn+5]
                old_v[idx*3+1] = quad_state_list[-1][idx*self.dim_xn+6]
                old_target_v[0] = target_state_list[-1][2]
                old_target_v[1] = target_state_list[-1][3]

        x = np.zeros((self.num_agents, self.dim_xn))

        for idx in range(self.num_agents):
            x[idx,0] = agents_position_list[idx][0]
            x[idx,1] = agents_position_list[idx][1]
            x[idx,2] = agents_position_list[idx][2]
            x[idx,3] = old_v[idx*2]
            x[idx,4] = old_v[idx*2+1]
            x[idx,5] = old_v[idx*2+2]

        TargetObserved = [target_position[0], target_position[1], 0, old_target_v[0], old_target_v[1], 0, 0, 0, 0, 0, 0, 0]

        quad_state = np.zeros((self.num_agents*self.dim_xn+1))
        quad_state[0] = time.time() - time_begin
        for idx in range(self.num_agents):
            u = -np.matmul(self.K, x[idx] - self.z[idx] - TargetObserved)
            u = u.tolist()[0]
            u[0] = u[0] + self.m*self.g

            newX = self.MyQuad._discDynFun(x[idx], u)
            
            quad_state[idx*self.dim_xn] = newX[0]
            quad_state[idx*self.dim_xn+1] = newX[1]
            quad_state[idx*self.dim_xn+2] = self.z[idx][2]
            
        quad_state_list = np.vstack((quad_state_list, quad_state))
        target_state_list = np.vstack((target_state_list, np.hstack((target_position[0], target_position[1], old_target_v[0], old_target_v[1]))))

        self.save_Traj(time_begin, quad_state_list, target_state_list)
        return quad_state_list, target_state_list
        

    def save_Traj(self, time_begin, quad_state_list, target_state_list):
        for idx in range(self.num_agents):
            # output trajectories as a CSV file
            array_csv = quad_state_list[:,0]
            array_csv = np.vstack((array_csv, quad_state_list[:,idx*6+1], quad_state_list[:,idx*6+2], quad_state_list[:,idx*6+3]))
            array_csv = np.vstack((array_csv, quad_state_list[:,idx*6+4], quad_state_list[:,idx*6+5], quad_state_list[:,idx*6+6]))
            filename_csv = os.path.expanduser("~") + "/tianyu/Mambo-Tracking-Interface" + self.config_data_list[idx]["DIRECTORY_TRAJ"] + self.time_name + ".csv"
            np.savetxt(filename_csv, array_csv, delimiter=",")
        
        array_csv = quad_state_list[:,0]
        array_csv = np.vstack((array_csv, target_state_list[:,0], target_state_list[:,1]))
        filename_csv = os.path.expanduser("~") + "/tianyu/Mambo-Tracking-Interface/scripts_aimslab/traj_csv_files/jackal/" + self.time_name + ".csv"
        np.savetxt(filename_csv, array_csv, delimiter=",")

    def get_desire_state(self, target_position):


        for idx in range(self.num_agents-1):
            if idx == 0:
                desire_state = np.array(self.distance[idx*int(self.dim_xn_single /2):(idx+1)*int(self.dim_xn_single /2)])
                desire_state = np.vstack((desire_state, np.zeros((int(self.dim_xn_single /2),1))))
            else:
                desire_state = np.vstack((desire_state, self.distance[idx*int(self.dim_xn_single /2):(idx+1)*int(self.dim_xn_single /2)]))
                desire_state = np.vstack((desire_state, np.zeros((int(self.dim_xn_single /2),1))))
        desire_state = np.vstack((desire_state, target_position[0], target_position[1]))
        desire_state = np.vstack((desire_state, np.zeros((int(self.dim_xn_single /2),1))))

        return desire_state

    def fill_diagonal(self, N_agent, D):
        row = len(D)
        col = len(D[0])
        An = np.zeros((row*N_agent, col*N_agent))
        for idx in range(N_agent):
            for j in range(row):
                for k in range(col):
                    An[idx*row+j][idx*col+k] = D[j][k]
        return An

    def linear_transformation(self):
        T = np.zeros((self.dim_xn, self.dim_xn))
        for idx in range(self.num_agents-1):
            for k in range(self.dim_xn_single ):
                T[idx*self.dim_xn_single +k][k] = -1
                T[idx*self.dim_xn_single +k][(idx+1)*self.dim_xn_single +k] = 1
        for idx in range(self.dim_xn_single ):
            for k in range(self.num_agents):
                T[(self.num_agents-1)*self.dim_xn_single +idx][k*self.dim_xn_single +idx] = 1/self.num_agents

        return T

    async def update_states_mocap(self):
        """
        Update the positions from motion capture system.

        Output:
            positions_list: 2D list for agents positions (in Qualisys coordinates),
                [[x0,y0,z0], [x1,y1,z1], ...]
        """
        positions_list = list()
        for idx in range(self.num_agents):
            msg = await self.protocol_states_list[idx].recvfrom()
            positions_list.append(np.frombuffer(msg, dtype=np.float64).tolist())
        return positions_list

    async def update_target_mocap(self):
        """
        Update the target positions from motion capture system.
        """
        msg = await self.protocol_loop_target.recvfrom()
        position_target = np.frombuffer(msg, dtype=np.float64)
        # print("position_obs")
        # print(position_obs.tolist())
        return position_target.tolist()

