#!/usr/bin/env python3
import os
import sys
import numpy as np
import asyncio
sys.path.append(os.getcwd()+'/src')
from HumanInterface import InputWaypoints
from FormationHuman import FormationHuman


# dictionary for configuration
# dt for Euler integration
configDict = {"dt": 0.1, "stepNumHorizon": 10, "startPointMethod": "zeroInput"}

buildFlag = True
saveFlag = False

iniJackal = [-3,0,0]
iniQuad = [[-3.5,0.5,0],[-3.5,1,0],[-3.5,-1,0]]

desire = [[0.25,0,1],[-0.25,0.5,1],[-0.25,-0.5,1]]
t_change = np.inf

space_limit = [[-4,4],[-2,2]]
initial = iniJackal[0:2]
final = [3,0]
obs_size = 0.5
obs_position = [[-1,-0.5,obs_size,obs_size], [0.3,-0.7,obs_size,obs_size], [0.2,0.50,obs_size,obs_size], 
                [-2,1.5,obs_size,obs_size], [2,1.8,obs_size,obs_size]]

waypoints = [[-0.7,0.5], [1.5,0.2], [3,0]]
# waypoints = [[-1.2,0.3], [0.0,1.6], [1.0,1.6], [1.7,0.1], [3,0]]

array_csv = waypoints[0]
for idx in range(len(waypoints)-1):
    array_csv = np.vstack((array_csv, waypoints[idx+1]))
filename_csv = "src/waypoints.csv"
np.savetxt(filename_csv, array_csv, delimiter=",")

MySim = FormationHuman(configDict, waypoints, iniJackal, iniQuad, desire, obs_size, obs_position, t_change, buildFlag, saveFlag)
asyncio.run(MySim.run())
