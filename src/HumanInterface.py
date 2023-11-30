#!/usr/bin/env python3
import os
import sys
import time
sys.path.append(os.getcwd()+'/lib')
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class InputWaypoints(object):

    def __init__(self, initial, final, space_limit, obs_position):

        self.initialState = initial
        self.finalState = final
        # the lab space limit [meter] in x-axis [x_min, x_max]
        self.space_limit_x = space_limit[0]
        # the lab space limit [meter] in y-axis [y_min, y_max]
        self.space_limit_y = space_limit[1]
        self.obs_position = obs_position


    def run(self):
        """
        Run this method to obtain human inputs.
        """
        #---------------------------------------------------------------#
        # for plot view
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(1, 1, 1)
        # set axis limits
        self.ax.set_xlim([self.space_limit_x[0], self.space_limit_x[1]])
        self.ax.set_ylim([self.space_limit_y[0], self.space_limit_y[1]])
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_aspect('equal')
        # plot lab sapce boundary
        self.ax.plot(self.space_limit_x, [self.space_limit_y[0],self.space_limit_y[0]], color='black')
        self.ax.plot(self.space_limit_x, [self.space_limit_y[1],self.space_limit_y[1]], color='black')
        self.ax.plot([self.space_limit_x[0],self.space_limit_x[0]], self.space_limit_y, color='black')
        self.ax.plot([self.space_limit_x[1],self.space_limit_x[1]], self.space_limit_y, color='black')

        # plot start and goal
        self.ax.scatter(self.initialState[0], self.initialState[1], color='green')
        self.ax.scatter(self.finalState[0], self.finalState[1], color='violet')

        for idx in range(len(self.obs_position)):
            obs = patches.Rectangle((self.obs_position[idx][0], self.obs_position[idx][1]), self.obs_position[idx][2], self.obs_position[idx][3], 
                                        linewidth=1, edgecolor='black', facecolor='black')
            self.ax.add_patch(obs)
        

        # set legends
        colors = ["green", "violet", "red"]
        marker_list = ["o", "o", "+"]
        labels = ["start", "goal", "waypoints"]
        def f(marker_type, color_type): return plt.plot([], [], marker=marker_type, color=color_type, ls="none")[0]
        handles = [f(marker_list[i], colors[i]) for i in range(len(labels))]
        # add legend about path
        handles.append(patches.Patch(color="red", alpha=0.75))
        self.ax.legend(handles, labels, loc="upper left", framealpha=1)

        plt.title('Select waypoints. Middle click to terminate.', fontweight ='bold')
        print("Click your waypoints in XOY plane, order from -x to +x, -y to +y.")
        print("Middle click to terminate ginput.")
        waypoints = plt.ginput(0, 0)
        print("Waypoints selection completed.")

        # plot waypoints
        for i in range(len(waypoints)):
            # print("Waypoint XOY" + str(i+1) + ": x = " + str(waypoints[i][0]) + ", y = " + str(waypoints[i][1]))
            self.ax.scatter(waypoints[i][0], waypoints[i][1], color="blue")


        plt.draw()
        # waypoints output
        waypoints_output = []
        points = []
#        print()
        for i in range(len(waypoints)):
            waypoints_output.append([round(waypoints[i][0], 3), round(waypoints[i][1], 3)])
            points.append([waypoints[i][0], waypoints[i][1]])
            print("Waypoints [x, y] [meter]: ", end=" ")
            print(waypoints_output[i])
        waypoints_output.append(self.finalState)

        plt.close()

        return waypoints_output
     
