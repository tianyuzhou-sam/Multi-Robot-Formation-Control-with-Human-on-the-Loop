#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

desire_all = [[[0.25,0,1],[-0.25,0.5,1],[-0.25,-0.5,1]],
             [[0,0.5,1],[-0.25,-0.5,1],[0.25,-0.5,1]]]
desire = desire_all[0]       
t_change = [15, np.inf]
dt = 0.1
timeNow = 0
change_index = 0

jackal_data = np.genfromtxt('jackal_long.csv', delimiter=',')
mambo_01_data = np.genfromtxt('mambo_01_long.csv', delimiter=',')
mambo_02_data = np.genfromtxt('mambo_02_long.csv', delimiter=',')
mambo_03_data = np.genfromtxt('mambo_03_long.csv', delimiter=',')
target_position = [-0.5,0.5, 0.8,-1.35, 2,0]

ref = np.zeros((9, len(jackal_data[0])))
for idx_agent in range(3):
    timeNow = 0
    change_index = 0
    desire = desire_all[0] 
    for idx in range(len(jackal_data[0])):
        ref[3*idx_agent][idx] = jackal_data[1][idx] + desire[idx_agent][0]
        ref[3*idx_agent+1][idx] = jackal_data[2][idx] + desire[idx_agent][1]
        ref[3*idx_agent+2][idx] = 0 + desire[idx_agent][2]
        timeNow += dt
        if timeNow >= t_change[change_index]:
            change_index += 1
            desire = desire_all[change_index]
            print(desire)

error = np.zeros((3, len(jackal_data[0])))
for idx in range(len(jackal_data[0])):
    error1 = np.sqrt((mambo_01_data[1][idx]-ref[0][idx])**2 + (mambo_01_data[2][idx]-ref[1][idx])**2 + (mambo_01_data[3][idx]-ref[2][idx])**2)
    error[0][idx] = error1
    error2 = np.sqrt((mambo_02_data[1][idx]-ref[3][idx])**2 + (mambo_02_data[2][idx]-ref[4][idx])**2 + (mambo_02_data[3][idx]-ref[5][idx])**2)
    error[1][idx] = error2
    error3 = np.sqrt((mambo_03_data[1][idx]-ref[6][idx])**2 + (mambo_03_data[2][idx]-ref[7][idx])**2 + (mambo_03_data[3][idx]-ref[8][idx])**2)
    error[2][idx] = error3


fig, ax = plt.subplots()
ax.plot(jackal_data[0], error[0], color="blue", label='Agent 1')
ax.plot(jackal_data[0], error[1], color="yellow", label='Agent 2')
ax.plot(jackal_data[0], error[2], color="red", linestyle="dashed", label='Agent 3')
ax.set_xlabel('time (s)')
ax.set_ylabel('distance error (m)')
ax.set_title('Error in Formation')
ax.legend()

plt.show()
