#!/usr/bin/env python3
import os
import sys
import numpy as np
import asyncio
sys.path.append(os.getcwd()+'/src')
from HumanInterface import InputWaypoints
from FormationSimMultiDynamic import FormationSim


# dictionary for configuration
# dt for Euler integration
configDict = {"dt": 0.1, "stepNumHorizon": 10, "startPointMethod": "zeroInput"}

buildFlag = True
saveFlag = False
multipleFormation = False

iniJackal = [-2,0,0]
iniQuad = [[-2.5,0.5,0],[-2.5,1,0],[-2.5,-1,0]]

desire = [[0.25,0,0.8],[-0.25,0.5,1],[-0.25,-0.5,1]]
t_change = np.inf

if multipleFormation:
    desire = [[[0.25,0,1],[-0.25,0.5,1],[-0.25,-0.5,1]],
              [[0,0.5,1],[-0.25,-0.5,1],[0.25,-0.5,1]]]       
    t_change = [15, np.inf]

obs_size = 0.26
obs_position = [[-0.65-obs_size,-0.28-5*obs_size,obs_size,5*obs_size], 
                        [0.73,-0.60,obs_size,5*obs_size]]
    
space_limit = [[-2.5,2.5],[-2,2]]
initial = iniJackal[0:2]
final = [2,0]
obs_size = 0.26
obs_position = [[-0.65-obs_size,-0.28-5*obs_size,obs_size,5*obs_size], [0.73,-0.60,obs_size,5*obs_size]]

# waypoints = [[3,0]]
# Input = InputWaypoints(initial, final, space_limit, obs_position, waypoints)
# waypoints = Input.run()
waypoints = [[-0.5,0.5], [0.8,-1.35], [2,0]]
MySim = FormationSim(configDict, waypoints, iniJackal, iniQuad, desire, obs_size, obs_position, t_change, buildFlag, saveFlag)
asyncio.run(MySim.run())
