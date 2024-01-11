#!/usr/bin/env python3
import os
import sys
import asyncio
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import mpl_toolkits.mplot3d.art3d as art3d
sys.path.append(os.getcwd()+'/src')
from plot_traj import Simulator

async def run_simulator():
    
    resolution = 2
    value_non_obs = 0
    value_obs = 255

    simulationFlag = False
    
    if simulationFlag:
        map_width_meter = 8
        map_height_meter = 6
        map_center = [0,0.5]
        # jackal_data = np.genfromtxt('data/jackal_pre.csv', delimiter=',')
        # mambo_01_data = np.genfromtxt('data/mambo_01_pre.csv', delimiter=',')
        # mambo_02_data = np.genfromtxt('data/mambo_02_pre.csv', delimiter=',')
        # mambo_03_data = np.genfromtxt('data/mambo_03_pre.csv', delimiter=',')
        # target_position = [-1.2,0.3, 1.7,0.1, 3,0]
        jackal_data = np.genfromtxt('data/jackal_long.csv', delimiter=',')
        mambo_01_data = np.genfromtxt('data/mambo_01_long.csv', delimiter=',')
        mambo_02_data = np.genfromtxt('data/mambo_02_long.csv', delimiter=',')
        mambo_03_data = np.genfromtxt('data/mambo_03_long.csv', delimiter=',')
        target_position = [-1.2,0.3, 0.0,1.6, 1.0,1.6, 1.7,0.1, 3,0]
        obs_size = 0.5
        obs_position = [[-1,-0.5,obs_size,obs_size], [0.3,-0.7,obs_size,obs_size], [0.2,0.50,obs_size,obs_size], 
                [-2,1.5,obs_size,obs_size], [2,1.8,obs_size,obs_size]]

    else:
        map_width_meter = 6
        map_height_meter = 4
        map_center = [0,0]
        jackal_data = np.genfromtxt('experiment/traj_records/jackal.csv', delimiter=',')
        mambo_01_data = np.genfromtxt('experiment/traj_records/mambo_01.csv', delimiter=',')
        mambo_02_data = np.genfromtxt('experiment/traj_records/mambo_02.csv', delimiter=',')
        mambo_03_data = np.genfromtxt('experiment/traj_records/mambo_03.csv', delimiter=',')
        obs_size = 0.26
        obs_position = [[-0.65-obs_size,-0.28-5*obs_size,obs_size,5*obs_size], [0.73,-0.60,obs_size,5*obs_size]]
        target_position = [-0.5,0.25, 0.9,-1.3, 1.9,0.0]

    timeTraj = []
    jackal_position = []
    mambo_01_position = []
    mambo_02_position = []
    mambo_03_position = []

    for idx in range(len(jackal_data[0])):
        timeTraj.append(jackal_data[0][idx])
        jackal_position.append(jackal_data[1][idx])
        jackal_position.append(jackal_data[2][idx])
        mambo_01_position.append(mambo_01_data[1][idx])
        mambo_01_position.append(mambo_01_data[2][idx])
        mambo_02_position.append(mambo_02_data[1][idx])
        mambo_02_position.append(mambo_02_data[2][idx])
        mambo_03_position.append(mambo_03_data[1][idx])
        mambo_03_position.append(mambo_03_data[2][idx])

    all_position = np.vstack((jackal_position, mambo_01_position, mambo_02_position, mambo_03_position))

    agent_position = []
    agent_position.append(jackal_position[0])
    agent_position.append(jackal_position[1])
    agent_position.append(mambo_01_position[0])
    agent_position.append(mambo_01_position[1])
    agent_position.append(mambo_02_position[0])
    agent_position.append(mambo_02_position[1])
    agent_position.append(mambo_03_position[0])
    agent_position.append(mambo_03_position[1])

    MySimulator = Simulator(map_width_meter, map_height_meter, map_center, resolution, value_non_obs, value_obs)



    ax = MySimulator.create_realtime_plot(realtime_flag=True, path_legend_flag=True, legend_flag=True)

    tdx = 0
    idx = 0

    while idx < len(jackal_data[0]-1):
        agent_position = []
        agent_position.append(jackal_position[idx*2])
        agent_position.append(jackal_position[idx*2+1])
        agent_position.append(mambo_01_position[idx*2])
        agent_position.append(mambo_01_position[idx*2+1])
        agent_position.append(mambo_02_position[idx*2])
        agent_position.append(mambo_02_position[idx*2+1])
        agent_position.append(mambo_03_position[idx*2])
        agent_position.append(mambo_03_position[idx*2+1])
        MySimulator.update_realtime_plot(all_position, agent_position, target_position, obs_position, ax, legend_flag=True)

        # time_str = "T = " + str(jackal_data[0][idx]) + "s"
        # plt.text(0.25, 0.9, time_str, fontsize=14, transform=plt.gcf().transFigure)
        plt.pause(1E-6)

        idx += 1

        await asyncio.sleep((jackal_data[0][idx+1]-jackal_data[0][idx])/2)


asyncio.run(run_simulator())