#!/usr/bin/env python3
import os
import sys
import numpy as np
sys.path.append(os.getcwd()+'/src')
from JackalSys import JackalSys

if __name__ == '__main__':
    # dictionary for configuration
    # dt for Euler integration
    configDict = {"dt": 0.1}

    buildFlag = True
    # buildFlag = False

    # initialize JackalSys
    MyJackal = JackalSys(configDict=configDict, buildFlag=buildFlag)

    x0 = np.array([0, 0, 0])
    u0= np.array([1.0, 0.0])

    x0_dot = MyJackal._contDynFun(x0, u0)
    x1 = MyJackal._discDynFun(x0, u0)
    print("x0_dot: ", x0_dot)
    print("x1: ", x1)

