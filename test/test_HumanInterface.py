#!/usr/bin/env python3
import os
import sys
import numpy as np
sys.path.append(os.getcwd()+'/src')
from HumanInterface import InputWaypoints

if __name__ == '__main__':

    space_limit = [[-2.5,2.5],[-2,2]]
    initial = [-2,0]
    final = [2,0]
    obs_size = 0.5
    obs_position = [[-1,-0.5,obs_size,obs_size], [0.3,-0.7,obs_size,obs_size], [0.2,0.50,obs_size,obs_size]]
    Input = InputWaypoints(initial, final, space_limit, obs_position)
    waypoints = Input.run()

