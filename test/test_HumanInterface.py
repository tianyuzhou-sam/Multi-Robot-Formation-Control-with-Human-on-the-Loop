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
    Input = InputWaypoints(initial, final, space_limit)
    waypoints = Input.run()
