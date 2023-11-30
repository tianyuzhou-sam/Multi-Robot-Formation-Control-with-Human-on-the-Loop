#!/usr/bin/env python3
import os
import sys
import numpy as np
sys.path.append(os.getcwd()+'/src')
from HumanInterface import InputWaypoints
from FormationSim import FormationSim


# dictionary for configuration
# dt for Euler integration
configDict = {"dt": 0.1, "stepNumHorizon": 10, "startPointMethod": "zeroInput"}

buildFlag = True
saveFlag = False

iniJackal = [-2,0,0]
iniQuad = [[-2.5,0.5,0],[-2.5,1,0],[-2.5,-1,0]]
desire = [[0.25,0,1],[-0.25,0.5,1],[-0.25,-0.5,1]]

space_limit = [[-2.5,2.5],[-2,2]]
initial = iniJackal[0:2]
final = [2,0]
obs_size = 0.26
obs_position = [[-0.65-obs_size,-0.28-5*obs_size,obs_size,5*obs_size], [0.73,-0.60,obs_size,5*obs_size]]
Input = InputWaypoints(initial, final, space_limit, obs_position)
waypoints = Input.run()
waypoints = [[-0.5,0.5], [0.8,-1.35], [2,0]]
MySim = FormationSim(configDict, waypoints, iniJackal, iniQuad, desire, buildFlag, saveFlag)
MySim.run()
