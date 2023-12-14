#!/usr/bin/env python3
import numpy as np
import asyncio
import os
import sys
sys.path.append(os.getcwd()+'/experiment/src')
from ModelPredictiveControlOnLoop import ModelPredictiveControlOnLoop

if __name__ == '__main__':
    # dictionary for configuration
    # dt for Euler integration
    configDict = {"dt": 0.1, "stepNumHorizon": 5, "startPointMethod": "zeroInput"}
    config_file_name = 'experiment/config/config_jackal.json'

    buildFlag = True
    saveFlag = False

    x0 = np.array([-1.5, 0, 0])
    u0 = np.array([0, 0])
    T = 60

    # obstacle: -0.65, -0.3    5 boxes-y
    #            0.73, -0.67   5 boxes+y
    targets = [[-1.5, 0, 0], [1.9,0.0]]

    # initialize MPC
    MyMPC = ModelPredictiveControlOnLoop(configDict, buildFlag, targets, saveFlag, config_file_name)

    # Run our asynchronous main function forever
    asyncio.ensure_future(MyMPC.run(x0, T))
    asyncio.get_event_loop().run_forever()