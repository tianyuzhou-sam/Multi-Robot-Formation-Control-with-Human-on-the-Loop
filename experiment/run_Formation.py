#!/usr/bin/env python3
import numpy as np
import asyncio
import os
import sys
sys.path.append(os.getcwd()+'/experiment/src')
from Formation import FormationPlanner

if __name__ == "__main__":
    # initialize a planner with Motion Capture System for multiple agents
    MyPlanner = FormationPlanner()

    # formation = np.array([[-1.2],[1.0],[-1.2],[-1.0]]) run 7
    # formation = np.array([[-1.5],[1.0],[-1.5],[-1.0]]) run 8 
    # formation = np.array([[-1.5],[0.9],[-1.5],[-0.9]]) run 9 10
    formation = np.array([[-1.5],[0.9],[-1.5],[-0.9]])
    # run the planner online
    asyncio.run(MyPlanner.run_planner(formation))
    asyncio.ensure_future(MyPlanner.run_planner(formation))
