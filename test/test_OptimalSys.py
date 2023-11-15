#!/usr/bin/env python3
import os
import sys
import numpy as np
sys.path.append(os.getcwd()+'/src')
from JackalSys import JackalSys
from OptimalControlProblem import OptimalControlProblem

if __name__ == '__main__':
    # dictionary for configuration
    # dt for Euler integration
    configDict = {"dt": 0.1, "stepNumHorizon": 10, "startPointMethod": "zeroInput"}

    buildFlag = True
    # buildFlag = False

    # initialize JackalSys
    MyJackal = JackalSys(configDict, buildFlag)
    target = [5, 5]

    # initialize OptimalControlProblem
    MyOC = OptimalControlProblem(configDict, MyJackal, buildFlag)

    # test
    x0 = np.array([0, 0, 0])
    u0 = np.array([0, 0])
    decisionAll = MyOC._computeStartingPoint(x0)

    # solve
    xTraj, uTraj, timeTraj, ipoptTime, returnStatus, successFlag = MyOC.solve(x0, timeNow=0.0, target=target)
    print(xTraj)
    print(uTraj)