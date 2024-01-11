#!/usr/bin/env python3
import os
import sys
import numpy as np
sys.path.append(os.getcwd()+'/src')
from HumanInterface import InputWaypoints
from ModelPredictiveControl import ModelPredictiveControl

if __name__ == '__main__':
    # dictionary for configuration
    # dt for Euler integration
    configDict = {"dt": 0.1, "stepNumHorizon": 10, "startPointMethod": "zeroInput"}

    buildFlag = True
    saveFlag = False

    x0 = np.array([-2, 0, 0])
    u0 = np.array([0, 0])

    space_limit = [[-2.5,2.5],[-2,2]]
    initial = x0[0:2]
    final = [2,0]
    waypoints = [final]
    obs_size = 0.26
    obs_position = [[-0.65-obs_size,-0.28-5*obs_size,obs_size,5*obs_size], [0.73,-0.60,obs_size,5*obs_size]]

    humanInterface = InputWaypoints(initial, final, space_limit, obs_position, waypoints)
    waypoints = humanInterface.run()

    T = 45

    # initialize MPC
    MyMPC = ModelPredictiveControl(configDict, buildFlag, waypoints, saveFlag)
    result = MyMPC.run(x0, T)
    print(result)
    MyMPC.visualize(result)