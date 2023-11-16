#!/usr/bin/env python3
import numpy as np
import asyncio
import os
import sys
sys.path.append(os.getcwd()+'/experiment/lib')
from ModelPredictiveControl import ModelPredictiveControl

if __name__ == '__main__':
    # dictionary for configuration
    # dt for Euler integration
    configDict = {"dt": 0.1, "stepNumHorizon": 5, "startPointMethod": "zeroInput"}
    config_file_name = 'experiment/config/config_jackal.json'

    buildFlag = True
    saveFlag = False

    x0 = np.array([0, 0, 0])
    u0 = np.array([0, 0])
    T = 60
    targets = [[-0.,0.], [1,-0.5], [2,0]]

    # initialize MPC
    MyMPC = ModelPredictiveControl(configDict, buildFlag, targets, saveFlag, config_file_name)

    # Run our asynchronous main function forever
    asyncio.ensure_future(MyMPC.run(x0, T))
    asyncio.get_event_loop().run_forever()