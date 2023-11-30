#!/usr/bin/env python3
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from random import randint


class Simulator:
    resolution: int
    map_width: int
    map_height: int
    map_center: list
    value_non_obs: int
    value_obs: int
    size_obs_width: int
    size_obs_height: int

    def __init__(self, 
        map_width_meter: float,
        map_height_meter: float,
        map_center: list,
        resolution: int,
        value_non_obs: int,
        value_obs: int):
        """
        Constructor
        
        the cell is empty if value_non_obs, the cell is blocked if value_obs.
        """
        # map resolution, how many cells per meter
        self.resolution = resolution
        self.map_width_meter = map_width_meter
        self.map_height_meter = map_height_meter
        # how many cells for width and height
        map_width = map_width_meter * resolution + 1
        map_height = map_height_meter * resolution + 1

        self.map_width = int(map_width)
        self.map_height = int(map_height)
        self.map_center = map_center

        self.value_non_obs = value_non_obs
        self.value_obs = value_obs

        # create an empty map
        self.map_array = np.array([self.value_non_obs]*(self.map_width*self.map_height)).reshape(-1, self.map_width)

    def position_to_map_index(self, position: list):
        """
        Given the position in meter [px, py], return the index of the map associated with this position.
        Positive-X points right, positive-Y points forwards.

        Input:
            position: a 1D list for position in meter [px, py]

        Output:
            idx_list: a 1D list [idx_column, idx_row] for index of width and height in self.map_array
        """
        idx_column = int( position[0] * self.resolution )
        idx_row = int( position[1] * self.resolution )
        if (idx_column > self.map_width-1) or (idx_column < 0) or (idx_row > self.map_height-1) or (idx_row < 0):
            raise Exception("Position in x-axis or y-axis exceed the range!")

        return [idx_column, idx_row]

    def map_index_to_position(self, map_index: list):
        """
        Given the map index list [idx_column, idx_row], return the position in meter [px, py] associated with this index list.
        Positive-X points right, positive-Y points forwards.

        Input:
            map_index: a 1D list [idx_column, idx_row] for index of width and height in self.map_array

        Output:
            position: a 1D list for position in meter [px, py]
        """
        return [map_index[0]/self.resolution, map_index[1]/self.resolution]

    def plot_single_path(self, path_single: list):
        """
        Plot a single path.

        Input:
            path_single: a 1D list for the path [x0,y0, x1,y1, x2,y2, ...]
        """
        if path_single:
            ax_map = self.create_realtime_plot(realtime_flag=False, path_legend_flag=True)
            # plot the map
            cmap = matplotlib.colors.ListedColormap(['white','black'])
            ax_map.pcolormesh(self.map_array, cmap=cmap, edgecolors='none')
            ax_map.scatter(path_single[0], path_single[1], marker="o", color="blue")
            ax_map.scatter(path_single[-2], path_single[-1], marker="x", color="red")
            ax_map.plot(list(map(lambda x:x, path_single[0::2])),
                        list(map(lambda x:x, path_single[1::2])), color="green", linewidth=2)
            plt.show(block=False)
        else:
            print("No path!")

    def plot_paths(self, path_many_agents: list, agents_position: list,
                   targets_position: list, obs_position: list, legend_flag=True, 
                   agent_text_flag=True, target_text_flag=True,
                   blockFlag=False, plotFirstFigFlag=False):
        """
        Plot many paths for multiple agents.

        Input:
            path_many_agents: a 3D list for paths, each 2D list is the path for each agent,
                [[x0,y0, x1,y1, x2,y2, ...], [x0,y0, x1,y1, x2,y2, ...], ...]
            agents_position: a 1D list for all the agent position, [x0,y0, x1,y1, x2,y2, ...]
            targets_position: a 1D list for all the target position, [x0,y0, x1,y2, x2,y2, ...]
            obs_position: left-bottom corner, width and height, [[x0,y0,w0,h0], [x1,y1,w1,h1], ...]
        """
        # offset to plot text in 2D space
        text_offset = [-0.2,0.]

        if plotFirstFigFlag:
            # the first figure is without the solved path
            realtime_flag = False
            cluster_legend_flag = False
            path_legend_flag = False
            ax_before = self.create_realtime_plot(realtime_flag, path_legend_flag, legend_flag)
            # plot the map
            cmap = matplotlib.colors.ListedColormap(['white','black'])
            ax_before.pcolormesh(self.map_array, cmap=cmap, edgecolors='none')
            # plot agents and targets
            self.plot_agents(agents_position, text_offset, ax_before, agent_text_flag)
            self.plot_targets(targets_position, text_offset, ax_before, target_text_flag)

        # the second figure is with the solved path
        realtime_flag = False
        cluster_legend_flag = True
        path_legend_flag = True
        ax = self.create_realtime_plot(realtime_flag, path_legend_flag, legend_flag)
        # plot the map
        cmap = matplotlib.colors.ListedColormap(['white','black'])
        ax.pcolormesh(self.map_array, cmap=cmap, edgecolors='none')
        # plot agents and targets
        self.plot_agents(agents_position, text_offset, ax, agent_text_flag)
        self.plot_targets(targets_position, text_offset, ax, target_text_flag)
        # plot paths
        self.plot_paths_figure(path_many_agents, agents_position, targets_position, ax)
        if obs_position is not None:
            for idx in range(len(obs_position)):
                obs = patches.Rectangle((obs_position[idx][0], obs_position[idx][1]), obs_position[idx][2], obs_position[idx][3], 
                                         linewidth=1, edgecolor='black', facecolor='black')
                ax.add_patch(obs)

        plt.show(block=blockFlag)

    
    def create_realtime_plot(self, realtime_flag=True, path_legend_flag=True, legend_flag=True):
        """
        Create a realtime plotting figure.
        """
        _, ax = plt.subplots(1, 1, figsize=(12, 10))
        if realtime_flag:
            plt.ion()
            plt.show()
        # figure settings
        handles, labels = self.figure_settings(ax, path_legend_flag, legend_flag)
        if legend_flag:
            legend = plt.legend(handles, labels, bbox_to_anchor=(1, 1), loc="upper left", framealpha=1)
        return ax

    def update_realtime_plot(self, path_many_agents: list, agents_position: list,
                             targets_position: list, ax, legend_flag=True):
        """
        Update realtime plotting once in an existing figure. See input details in self.plot_paths()
        """
        ax.clear()  # clear the previous figure

        # offset to plot text in 2D space
        text_offset = (self.map_width - 0) / 40
        cmap = matplotlib.colors.ListedColormap(['white','black'])
        # plot the map
        ax.pcolormesh(self.map_array, cmap=cmap, alpha=1.0, edgecolors='none')
        # plot agents and targets
        self.plot_agents(agents_position, text_offset, ax)
        self.plot_targets(targets_position, text_offset, ax)
        if path_many_agents:
            # plot paths
            self.plot_paths_figure(path_many_agents, agents_position,
                                   targets_position, ax)
        if legend_flag:
            # plot legends
            handles, labels = self.figure_settings(ax, path_legend_flag=True)
            legend = plt.legend(handles, labels, bbox_to_anchor=(1, 1), loc="upper left", framealpha=1)
        plt.draw()

    def plot_agents(self, agents_position: list, text_offset: list, ax, agent_text_flag=True):
        """
        Plot agents.
        """
        for idx_agent in range(int(len(agents_position)/2)):
            if idx_agent == 0:
                agent_color = "red"
            else:
                agent_color = "blue"
            ax.scatter(agents_position[2*idx_agent], agents_position[2*idx_agent+1], marker="o", color=agent_color)
            if agent_text_flag:
                ax.text(agents_position[2*idx_agent]+text_offset[0], agents_position[2*idx_agent+1]+text_offset[1],
                        "A"+str(idx_agent), fontweight="bold", color=agent_color)

    def plot_targets(self, targets_position: list, text_offset: list, ax, target_text_flag=True):
        """
        Plot targets.
        """
        
        # if cluster_centers is empty, plot targets
        for idx_target in range(int(len(targets_position)/2)):
            ax.scatter(targets_position[2*idx_target], targets_position[2*idx_target+1], s=100, marker="*", color="red")
            if target_text_flag:
                ax.text(targets_position[2*idx_target]+text_offset[0], targets_position[2*idx_target+1]+text_offset[1],
                        "T"+str(idx_target), fontweight="bold", color="red")

    def plot_paths_figure(self, path_many_agents: list, agents_position: list,
                          targets_position: list, ax):
        """
        Plot path for multiple agents in an existing figure.
        """

        for idx_agent in range(len(path_many_agents)):
            path_many_each_agent = path_many_agents[idx_agent]

            for idx_path in range(len(path_many_each_agent)):
                if idx_agent == 0:
                    agent_color = "red"
                else:
                    agent_color = "blue"
                ax.plot(list(map(lambda x:x, path_many_each_agent[0::2])),
                        list(map(lambda x:x, path_many_each_agent[1::2])),
                        linewidth=2, color=agent_color, linestyle="solid")
                

    def figure_settings(self, ax, path_legend_flag: bool, legend_flag=True):
        """
        Settings for the figure.

        path_legend_flag = True if plot path legends
        """
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title("T = 25s", fontweight='bold')
        ax.set_xlim([self.map_center[0]-self.map_width_meter/2, self.map_center[0]+self.map_width_meter/2])
        ax.set_ylim([self.map_center[1]-self.map_height_meter/2, self.map_center[1]+self.map_height_meter/2])

        # set legends
        if legend_flag:
            colors = ["red", "blue", "red"]
            marker_list = ["o", "o", "*"]
            labels = ["Leader", "Agent", "Waypoint"]
            f = lambda m,c: plt.plot([],[],marker=m, color=c, ls="none")[0]
            handles = [f(marker_list[i], colors[i]) for i in range(len(labels))]

            if path_legend_flag:
                # add legend about path
                handles.append(plt.plot([],[], linestyle="solid", color="red", linewidth=2)[0])
                handles.append(plt.plot([],[], linestyle="solid", color="blue", linewidth=2)[0])
                handles.append(plt.plot([],[], marker="s", color="black", ls="none")[0])
                labels.extend(["Path", "Path", "Obstacles"])
                # a tuple includes the handles and labels of legend
        else:
            handles = list()
            labels = list()
        return handles, labels


if __name__ == '__main__':
    map_width_meter = 6
    map_height_meter = 4
    map_center = [0,0]
    resolution = 2
    value_non_obs = 0
    value_obs = 255
    obs_size = 0.26

    simulationFlag = True

    agent_position = [0,0, 0,0, 0,0, 0,0]
    obs_position = [[-0.65-obs_size,-0.28-5*obs_size,obs_size,5*obs_size], [0.73,-0.60,obs_size,5*obs_size]]

    

    if simulationFlag:
        jackal_data = np.genfromtxt('jackal.csv', delimiter=',')
        mambo_01_data = np.genfromtxt('mambo_01.csv', delimiter=',')
        mambo_02_data = np.genfromtxt('mambo_02.csv', delimiter=',')
        mambo_03_data = np.genfromtxt('mambo_03.csv', delimiter=',')
        target_position = [-0.5,0.5, 0.8,-1.35, 2,0]
    else:
        jackal_data = np.genfromtxt('experiment/traj_records/jackal.csv', delimiter=',')
        mambo_01_data = np.genfromtxt('experiment/traj_records/mambo_01.csv', delimiter=',')
        mambo_02_data = np.genfromtxt('experiment/traj_records/mambo_02.csv', delimiter=',')
        mambo_03_data = np.genfromtxt('experiment/traj_records/mambo_03.csv', delimiter=',')
        target_position = [-0.5,0.25, 0.9,-1.3, 1.9,0.0]

    jackal_position = []
    mambo_01_position = []
    mambo_02_position = []
    mambo_03_position = []

    case = 0

    if case == 0:
        plotLength = len(jackal_data[0])        # 25s
    elif case == 1:
        plotLength = 1          # 0s
    elif case == 2:
        plotLength = 50         # 5s
    elif case == 3:
        plotLength = 70         # 7s   
    elif case == 4:
        plotLength = 100        # 10s 
    elif case == 5:
        plotLength = 200        # 20s   


    for idx in range(plotLength):
        jackal_position.append(jackal_data[1][idx])
        jackal_position.append(jackal_data[2][idx])
        mambo_01_position.append(mambo_01_data[1][idx])
        mambo_01_position.append(mambo_01_data[2][idx])
        mambo_02_position.append(mambo_02_data[1][idx])
        mambo_02_position.append(mambo_02_data[2][idx])
        mambo_03_position.append(mambo_03_data[1][idx])
        mambo_03_position.append(mambo_03_data[2][idx])

    agent_position = []
    agent_position.append(jackal_position[-2])
    agent_position.append(jackal_position[-1])
    agent_position.append(mambo_01_position[-2])
    agent_position.append(mambo_01_position[-1])
    agent_position.append(mambo_02_position[-2])
    agent_position.append(mambo_02_position[-1])
    agent_position.append(mambo_03_position[-2])
    agent_position.append(mambo_03_position[-1])

    MyPlot = Simulator(map_width_meter, map_height_meter, map_center, resolution, value_non_obs, value_obs)

    all_position = np.vstack((jackal_position, mambo_01_position, mambo_02_position, mambo_03_position))
    MyPlot.plot_paths(all_position, agent_position, target_position, obs_position)

    plt.show()
