#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

desire = [[0.25,0,1],[-0.25,0.5,1],[-0.25,-0.5,1]]
jackal_data = np.genfromtxt('jackal_1_long.csv', delimiter=',')
mambo_01_data = np.genfromtxt('mambo_01_1_long.csv', delimiter=',')
mambo_02_data = np.genfromtxt('mambo_02_1_long.csv', delimiter=',')
mambo_03_data = np.genfromtxt('mambo_03_1_long.csv', delimiter=',')

# desire = [[0.5,0,1],[-0.,0.75,1],[-0.,-0.75,1]]
# jackal_data = np.genfromtxt('experiment/traj/run18/jackal.csv', delimiter=',')
# mambo_01_data = np.genfromtxt('experiment/traj/run18/mambo_01.csv', delimiter=',')
# mambo_02_data = np.genfromtxt('experiment/traj/run18/mambo_02.csv', delimiter=',')
# mambo_03_data = np.genfromtxt('experiment/traj/run18/mambo_03.csv', delimiter=',')

target_position = [-0.5,0.5, 0.8,-1.35, 2,0]

ref = np.zeros((9, len(jackal_data[0])))
for idx_agent in range(3):
    for idx in range(len(jackal_data[0])):
        ref[3*idx_agent][idx] = jackal_data[1][idx] + desire[idx_agent][0]
        ref[3*idx_agent+1][idx] = jackal_data[2][idx] + desire[idx_agent][1]
        ref[3*idx_agent+2][idx] = 0 + desire[idx_agent][2]

error = np.zeros((3, len(jackal_data[0])))
for idx in range(len(jackal_data[0])):
    error1 = np.sqrt((mambo_01_data[1][idx]-ref[0][idx])**2 + (mambo_01_data[2][idx]-ref[1][idx])**2 + (mambo_01_data[3][idx]-ref[2][idx])**2)
    error[0][idx] = error1
    error2 = np.sqrt((mambo_02_data[1][idx]-ref[3][idx])**2 + (mambo_02_data[2][idx]-ref[4][idx])**2 + (mambo_02_data[3][idx]-ref[5][idx])**2)
    error[1][idx] = error2
    error3 = np.sqrt((mambo_03_data[1][idx]-ref[6][idx])**2 + (mambo_03_data[2][idx]-ref[7][idx])**2 + (mambo_03_data[3][idx]-ref[8][idx])**2)
    error[2][idx] = error3


waypoints = [[-0.5,0.5], [0.8,-1.35], [2,0]]
ref_x = []
ref_y = []
epsilon = 0.01
reached = 0
for idx in range(len(jackal_data[0])):
    ref_x.append(waypoints[reached][0])
    ref_y.append(waypoints[reached][1])
    if ((jackal_data[1][idx] - waypoints[reached][0])**2 + (jackal_data[2][idx] - waypoints[reached][1])**2 <= epsilon**2):
        if reached < len(waypoints)-1:
            reached += 1


fig, ax = plt.subplots()
ax.plot(jackal_data[0], error[0], color="blue", label='Quadrotor 1')
ax.plot(jackal_data[0], error[1], color="red", label='Quadrotor 2')
ax.plot(jackal_data[0], error[2], color="black", linestyle="dashed", label='Quadrotor 3')
ax.set_xlabel('time (s)')
ax.set_ylabel('distance error (m)')
ax.set_title('Error in Formation')
ax.legend()

fig2, ax2 = plt.subplots(2,1)
ax2[0].plot(jackal_data[0], jackal_data[1], color='blue', label='Actual')
ax2[0].plot(jackal_data[0], ref_x, color='red', linestyle='dashed', label='Reference')
ax2[0].set_ylabel('X (m)')
ax2[0].set_title('Leader Trajectory')
ax2[0].legend()

ax2[1].plot(jackal_data[0], jackal_data[2], color='blue')
ax2[1].plot(jackal_data[0], ref_y, color='red', linestyle='dashed')
ax2[1].set_ylabel('Y (m)')
ax2[1].set_xlabel('time (s)')
plt.show()
