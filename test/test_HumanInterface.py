#!/usr/bin/env python3
import os
import sys
import numpy as np
sys.path.append(os.getcwd()+'/src')
from HumanInterface import InputWaypoints

if __name__ == '__main__':

    space_limit = [[-4,4],[-2.5,3.5]]
    initial = [-3,0]
    final = [3,0]
    obs_size = 0.5
    
    waypoints = [[-0.7,0.5], [1.5,0.2], [3,0]]
    timeIndex = [5,15,25]

    obs_position = [[-1,-0.5,obs_size,obs_size], [0.3,-0.7,obs_size,obs_size], [0.2,0.50,obs_size,obs_size], 
                [-2,1.5,obs_size,obs_size], [2,1.8,obs_size,obs_size]]

    
    Input = InputWaypoints(initial, final, space_limit, obs_position, waypoints)
    new_waypoints = Input.run()
