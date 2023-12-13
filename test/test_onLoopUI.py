#!/usr/bin/env python3
import os
import sys
import numpy as np
sys.path.append(os.getcwd()+'/src')
from HumanInterface import InputWaypoints

if __name__ == '__main__':

    space_limit = [[-4,4],[-2,2]]
    initial = [-3,0]
    final = [3,0]
    obs_size = 0.5
    
    waypoints = [[-0.7,0.5], [1.5,0.2], [3,0]]
    timeIndex = [5,15,25]

    obs_position = [[-1,-0.5,obs_size,obs_size], [0.3,-0.7,obs_size,obs_size], [0.2,0.50,obs_size,obs_size], 
                [-2,1.5,obs_size,obs_size], [2,1.8,obs_size,obs_size]]

    
    Input = InputWaypoints(initial, final, space_limit, obs_position, waypoints)
    new_waypoints = Input.run()

    print("Enter time index for new waypoints:")
    newIndex = [int(x) for x in input().split()]
    print(newIndex)
    
    for idx in range(len(newIndex)):
        for jdx in range(len(timeIndex)):
            if newIndex[idx] < timeIndex[jdx]:
                timeIndex.insert(jdx, newIndex[idx])
                waypoints.insert(jdx, new_waypoints[idx])
                break

    array_csv = waypoints[0]
    for idx in range(len(waypoints)-1):
        array_csv = np.vstack((array_csv, waypoints[idx+1]))
    filename_csv = "src/waypoints.csv"
    np.savetxt(filename_csv, array_csv, delimiter=",")
    
    print(waypoints)
    while True:
        Input = InputWaypoints(initial, final, space_limit, obs_position, waypoints)
        new_waypoints = Input.run()
        for idx in range(len(new_waypoints)-1):
            waypoints.append(new_waypoints[idx])
        
        print("Enter time index for new waypoints:")
        newIndex = [int(x) for x in input().split()]
        print(newIndex)
        
        for idx in range(len(newIndex)):
            for jdx in range(len(timeIndex)):
                if newIndex[idx] < timeIndex[jdx]:
                    timeIndex.insert(jdx, newIndex[idx])
                    waypoints.insert(jdx, new_waypoints[idx])
                    break

        array_csv = waypoints[0]
        for idx in range(len(waypoints)-1):
            array_csv = np.vstack((array_csv, waypoints[idx+1]))
        filename_csv = "src/waypoints.csv"
        np.savetxt(filename_csv, array_csv, delimiter=",")

