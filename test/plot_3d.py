#!/usr/bin/env python3
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import mpl_toolkits.mplot3d.art3d as art3d


map_x_meter = 8
map_y_meter = 4
map_center = [0,0.5]
resolution = 2
value_non_obs = 0
value_obs = 255
obs_size = 0.5

simulationFlag = True

agent_position = [0,0,0, 0,0,0, 0,0,0, 0,0,0]
obs_position = [[-1,-0.5,obs_size,obs_size], [0.3,-0.7,obs_size,obs_size], [0.2,0.50,obs_size,obs_size], 
                [-2,1.5,obs_size,obs_size], [2,1.8,obs_size,obs_size]]

if simulationFlag:
    jackal_data = np.genfromtxt('data/jackal.csv', delimiter=',')
    mambo_01_data = np.genfromtxt('data/mambo_01.csv', delimiter=',')
    mambo_02_data = np.genfromtxt('data/mambo_02.csv', delimiter=',')
    mambo_03_data = np.genfromtxt('data/mambo_03.csv', delimiter=',')
    preTraj = np.genfromtxt('data/jackal_pre.csv', delimiter=',')
    waypoints = [[-1.2,0.3], [0.0,1.6], [1.0,1.6], [1.7,0.1], [3,0]]
else:
    # jackal_data = np.genfromtxt('experiment/traj/run18/jackal.csv', delimiter=',')
    # mambo_01_data = np.genfromtxt('experiment/traj/run18/mambo_01.csv', delimiter=',')
    # mambo_02_data = np.genfromtxt('experiment/traj/run18/mambo_02.csv', delimiter=',')
    # mambo_03_data = np.genfromtxt('experiment/traj/run18/mambo_03.csv', delimiter=',')
    # waypoints = [[-0.5,0.25], [0.9,-1.3], [1.9,0.0]]

    # jackal_data_ref = np.genfromtxt('experiment/traj_records/jackal.csv', delimiter=',')
    # mambo_01_data_ref = np.genfromtxt('experiment/traj_records/mambo_01.csv', delimiter=',')
    # mambo_02_data_ref = np.genfromtxt('experiment/traj_records/mambo_02.csv', delimiter=',')
    # mambo_03_data_ref = np.genfromtxt('experiment/traj_records/mambo_03.csv', delimiter=',')

    jackal_data = np.genfromtxt('experiment/traj_records/jackal.csv', delimiter=',')
    mambo_01_data = np.genfromtxt('experiment/traj_records/mambo_01.csv', delimiter=',')
    mambo_02_data = np.genfromtxt('experiment/traj_records/mambo_02.csv', delimiter=',')
    mambo_03_data = np.genfromtxt('experiment/traj_records/mambo_03.csv', delimiter=',')
    waypoints = [[-0.5,0.25], [0.9,-1.3], [1.9,0.0]]

jackal_position = []
mambo_01_position = []
mambo_02_position = []
mambo_03_position = []
jackal_pre = []

mambo_01_position_ref = []
mambo_02_position_ref = []
mambo_03_position_ref = []

# obs_size = 0.26
# obs_position = [[-0.65-obs_size,-0.28-5*obs_size,obs_size,5*obs_size], [0.73,-0.60,obs_size,5*obs_size]]

plotLength = len(jackal_data[0])        # 25s
plotskip = 1

for idx in range(int(plotLength/plotskip)):
    jackal_position.append(jackal_data[1][idx*plotskip])
    jackal_position.append(jackal_data[2][idx*plotskip])
    jackal_position.append(0)
    mambo_01_position.append(mambo_01_data[1][idx*plotskip])
    mambo_01_position.append(mambo_01_data[2][idx*plotskip])
    mambo_01_position.append(mambo_01_data[3][idx*plotskip])
    mambo_02_position.append(mambo_02_data[1][idx*plotskip])
    mambo_02_position.append(mambo_02_data[2][idx*plotskip])
    mambo_02_position.append(mambo_02_data[3][idx*plotskip])
    mambo_03_position.append(mambo_03_data[1][idx*plotskip])
    mambo_03_position.append(mambo_03_data[2][idx*plotskip])
    mambo_03_position.append(mambo_03_data[3][idx*plotskip])

for idx in range(len(preTraj[0])):
    jackal_pre.append(preTraj[1][idx])
    jackal_pre.append(preTraj[2][idx])
    jackal_pre.append(0)

# for idx in range(plotLength):
#     mambo_01_position_ref.append(mambo_01_data_ref[1][idx])
#     mambo_01_position_ref.append(mambo_01_data_ref[2][idx])
#     mambo_01_position_ref.append(mambo_01_data_ref[3][idx])
#     mambo_02_position_ref.append(mambo_02_data_ref[1][idx])
#     mambo_02_position_ref.append(mambo_02_data_ref[2][idx])
#     mambo_02_position_ref.append(mambo_02_data_ref[3][idx])
#     mambo_03_position_ref.append(mambo_03_data_ref[1][idx])
#     mambo_03_position_ref.append(mambo_03_data_ref[2][idx])
#     mambo_03_position_ref.append(mambo_03_data_ref[3][idx])


agent_position = []
agent_position.append(jackal_position[-3])
agent_position.append(jackal_position[-2])
agent_position.append(jackal_position[-1])
agent_position.append(mambo_01_position[-3])
agent_position.append(mambo_01_position[-2])
agent_position.append(mambo_01_position[-1])
agent_position.append(mambo_02_position[-3])
agent_position.append(mambo_02_position[-2])
agent_position.append(mambo_02_position[-1])
agent_position.append(mambo_03_position[-3])
agent_position.append(mambo_03_position[-2])
agent_position.append(mambo_03_position[-1])

formation = [[agent_position[3], agent_position[6], agent_position[9], agent_position[3]], 
             [agent_position[4], agent_position[7], agent_position[10], agent_position[4]], 
             [agent_position[5], agent_position[8], agent_position[11], agent_position[5]]]
projection = [[agent_position[3], agent_position[6], agent_position[9], agent_position[3]], 
             [agent_position[4], agent_position[7], agent_position[10], agent_position[4]], 
             [0,0,0,0]]

ax = plt.figure().add_subplot(projection='3d')
ax.plot(jackal_position[0::3], jackal_position[1::3], jackal_position[2::3], color='red')
ax.plot(jackal_pre[0::3], jackal_pre[1::3], jackal_pre[2::3], color='red', linestyle='dashed')
ax.plot(mambo_01_position[0::3], mambo_01_position[1::3], mambo_01_position[2::3], color='blue')
ax.plot(mambo_02_position[0::3], mambo_02_position[1::3], mambo_02_position[2::3], color='blue')
ax.plot(mambo_03_position[0::3], mambo_03_position[1::3], mambo_03_position[2::3], color='blue')
# ax.plot(mambo_01_position_ref[0::3], mambo_01_position_ref[1::3], mambo_01_position_ref[2::3], color='green')
# ax.plot(mambo_02_position_ref[0::3], mambo_02_position_ref[1::3], mambo_02_position_ref[2::3], color='green')
# ax.plot(mambo_03_position_ref[0::3], mambo_03_position_ref[1::3], mambo_03_position_ref[2::3], color='green')
ax.scatter(agent_position[0], agent_position[1], agent_position[2], color='red', label='Leader')
ax.scatter(agent_position[3], agent_position[4], agent_position[5], color='blue', label='Quadrotors')
ax.scatter(agent_position[6], agent_position[7], agent_position[8], color='blue')
ax.scatter(agent_position[9], agent_position[10], agent_position[11], color='blue')
ax.plot(formation[0], formation[1], formation[2], color='y', label='Formation')
ax.plot(projection[0], projection[1], projection[2], color='g', label='Projection')
for idx in range(len(waypoints)-1):
    if idx == 0:
        ax.scatter(waypoints[idx][0], waypoints[idx][1], 0, color='red', marker='*', s=50, label='Waypoints')
    else:
        ax.scatter(waypoints[idx][0], waypoints[idx][1], 0, color='red', marker='*', s=50)
for idx in range(len(obs_position)):
    if idx == 1:
        obs = patches.Rectangle((obs_position[idx][0], obs_position[idx][1]), obs_position[idx][2], obs_position[idx][3], 
                                linewidth=1, edgecolor='black', facecolor='black', label='Obstacles')
    else:
        obs = patches.Rectangle((obs_position[idx][0], obs_position[idx][1]), obs_position[idx][2], obs_position[idx][3], 
                                linewidth=1, edgecolor='black', facecolor='black')
    ax.add_patch(obs)
    art3d.pathpatch_2d_to_3d(obs, z=0, zdir="z")
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_zlabel("z (m)")
# ax.set_title("Formation Experiment", fontweight='bold')
ax.set_xlim([map_center[0]-map_x_meter/2, map_center[0]+map_x_meter/2])
ax.set_ylim([map_center[1]-map_y_meter/2, map_center[1]+map_y_meter/2])
ax.set_zlim([0, 2])
ax.legend()

plt.show()
