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


# when distance between A and B < this number, we say A and B have same position
DISTANCE_THRESHOLD = 0.2

class FormationPlanner:
    num_agents: int  # number of agents
    dt: float  # time step
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
        fly_time = 60
        # formation
        self.distance = formation

        # dynamic (only for simulation enviornment setup)
        g = 9.81
        m = 0.2
        Ixx = 0.1
        Iyy = 0.1
        Izz = 0.1

        A_single = np.array([[0,0,0,1,0,0,0,0,0,0,0,0],
                            [0,0,0,0,1,0,0,0,0,0,0,0],
                            [0,0,0,0,0,1,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,-g,0,0,0,0],
                            [0,0,0,0,0,0,g,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,1,0,0],
                            [0,0,0,0,0,0,0,0,0,0,1,0],
                            [0,0,0,0,0,0,0,0,0,0,0,1],
                            [0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0]])

        B_single = np.array([[0,0,0,0],
                            [0,0,0,0],
                            [0,0,0,0],
                            [0,0,0,0],
                            [0,0,0,0],
                            [1/m,0,0,0],
                            [0,0,0,0],
                            [0,0,0,0],
                            [0,0,0,0],
                            [0,1/Ixx,0,0],
                            [0,0,1/Iyy,0],
                            [0,0,0,1/Izz]])
        # dimension of group matrices
        self.dim_xn_single = len(B_single)                          # single agent
        self.dim_xn = self.dim_xn_single*self.num_agents                # number of rows
        self.dim_un = len(B_single[0])*self.num_agents              # number of columns
        
        # set weight
        Q = np.eye(self.dim_xn)*1
        Q[(self.num_agents-1)*4][(self.num_agents-1)*4] = 1
        Q[(self.num_agents-1)*4+1][(self.num_agents-1)*4+1] = 1
        R = np.eye(self.dim_un)*0.1

        # get froup matrices
        A = self.fill_diagonal(self.num_agents, A_single)
        B = self.fill_diagonal(self.num_agents, B_single)

        # linear transformation
        T = self.linear_transformation()

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

        self.height_fly = 1  # a constant fly height in meter
        self.time_name = time.strftime("%Y%m%d%H%M%S")

        time_begin = time.time()
        time_used = 0  # initialize the global time as 0
        quad_state_list = time.time() - time_begin
        for idx in range(self.num_agents):
            quad_state_list = np.hstack((quad_state_list, agents_position_list[idx][0:2], self.height_fly, 0, 0, 0))
        jackal_state_list = np.hstack((target_position[0], target_position[1]))

        A0 = np.matmul(np.matmul(T, A), np.linalg.inv(T))
        B0 = np.matmul(T, B)
        X0 = np.matrix(scipy.linalg.solve_continuous_are(A0, B0, Q, R))
        K = np.matrix(scipy.linalg.inv(R)*(B0.T*X0))

        time_start_fly = time.time()
        while(time_used < fly_time):
            t_start = time.time()

            # update the agent position
            agents_position_list = await self.update_states_mocap()

            # update target position
            target_position = await self.update_target_mocap()

            # do formation
            quad_state_list, jackal_state_list, states = self.formation(K, time_begin, agents_position_list, jackal_state_list, target_position, quad_state_list, A, B, T)

            # plt.pause(1E-9)
            time_sleep = max(0, self.dt - time.time() + t_start)
            time_used = time.time() - time_start_fly
            print("Current Time [sec]: " + str(time.time()-time_begin))
            await asyncio.sleep(time_sleep)

    def formation(self, K, time_begin, agents_position_list, jackal_state_list, target_position, quad_state_list, A, B, T):
        old_v = np.zeros((2*self.num_agents,1))
        for idx in range(self.num_agents):
            if quad_state_list.size == self.num_agents*6+1:
                old_v[idx*2] = quad_state_list[idx*6+4]
                old_v[idx*2+1] = quad_state_list[idx*6+5]
            else:
                old_v[idx*2] = quad_state_list[-1][idx*6+4]
                old_v[idx*2+1] = quad_state_list[-1][idx*6+5]
        
        x = np.array([[agents_position_list[0][0]],[agents_position_list[0][1]],[self.height_fly]])
        x = np.vstack((x, np.zeros((9,1))))
        for idx in range(self.num_agents-1):
            x = np.vstack((x, agents_position_list[idx+1][0], agents_position_list[idx+1][1], [self.height_fly]))
            x = np.vstack((x, np.zeros((9,1))))
        desire_state = self.get_desire_state(target_position[:2])

        z = np.matmul(T, x)
        X = z - desire_state
        u = -np.matmul(K, X)

        dx = np.matmul(A, x) + np.matmul(B, u)

        print(dx)
        x = x + dx*self.dt

        
        v = np.zeros((2*self.num_agents,1))
        for idx in range(self.num_agents):
            if quad_state_list.size == self.num_agents*6+1:
                v[idx*2] = (x.tolist()[idx*self.dim_xn_single]-quad_state_list[idx*6+1])/self.dt
                v[idx*2+1] = (x.tolist()[idx*self.dim_xn_single+1]-quad_state_list[idx*6+2])/self.dt
            else:
                v[idx*2] = (x.tolist()[idx*self.dim_xn_single]-quad_state_list[-1][idx*6+1])/self.dt
                v[idx*2+1] = (x.tolist()[idx*self.dim_xn_single+1]-quad_state_list[-1][idx*6+2])/self.dt

        quad_state = time.time() - time_begin
        for k in range(self.num_agents):
            quad_state = np.hstack((quad_state, x.tolist()[self.dim_xn_single*k], x.tolist()[self.dim_xn_single*k+1], self.height_fly, 0, 0, 0))
        quad_state_list = np.vstack((quad_state_list, quad_state))

        jackal_state_list = np.vstack((jackal_state_list, np.hstack((target_position[0], target_position[1]))))

        self.save_Traj(time_begin, quad_state_list, jackal_state_list)
        return quad_state_list ,jackal_state_list, x
        
    def save_Traj(self, time_begin, quad_state_list, jackal_state_list):
        for idx in range(self.num_agents):
            # output trajectories as a CSV file
            array_csv = quad_state_list[:,0]
            
            array_csv = np.vstack((array_csv, quad_state_list[:,idx*6+1], quad_state_list[:,idx*6+2], quad_state_list[:,idx*6+3]))
            array_csv = np.vstack((array_csv, quad_state_list[:,idx*6+4], quad_state_list[:,idx*6+5], quad_state_list[:,idx*6+6]))
            
            filename_csv = os.path.expanduser("~") + "/tianyu/Mambo-Tracking-Interface" + self.config_data_list[idx]["DIRECTORY_TRAJ"] + self.time_name + ".csv"
            np.savetxt(filename_csv, array_csv, delimiter=",")

        array_csv = quad_state_list[:,0]
        array_csv = np.vstack((array_csv, jackal_state_list[:,0], jackal_state_list[:,1]))
        filename_csv = os.path.expanduser("~") + "/tianyu/Mambo-Tracking-Interface/scripts_aimslab/traj_csv_files/jackal/" + self.time_name + ".csv"
        np.savetxt(filename_csv, array_csv, delimiter=",")

    def get_desire_state(self, target_position):
        for idx in range(self.num_agents-1):
            if idx == 0:
                desire_state = np.array(self.distance[0:2])
                desire_state = np.vstack((desire_state, np.zeros((self.dim_xn_single-2,1))))
            else:
                desire_state = np.vstack((desire_state, self.distance[2*idx:2*idx+2]))
                desire_state = np.vstack((desire_state, np.zeros((self.dim_xn_single-2,1))))
        desire_state = np.vstack((desire_state, target_position[0], target_position[1], self.height_fly))
        desire_state = np.vstack((desire_state, np.zeros((self.dim_xn_single-3,1))))

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

