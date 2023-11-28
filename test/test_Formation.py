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
iniQuad = [[-2,1,0],[-2,-1,0],[1,0,0]]
desire = [[-0.25,0.5,1],[-0.25,-0.5,1],[0.25,0,1]]

space_limit = [[-2.5,2.5],[-2,2]]
initial = iniJackal[0:2]
final = [2,0]
Input = InputWaypoints(initial, final, space_limit)
waypoints = Input.run()

MySim = FormationSim(configDict, waypoints, iniJackal, iniQuad, desire, buildFlag, saveFlag)
MySim.run()
