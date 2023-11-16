#!/usr/bin/env python3
import asyncio
import os
import sys
sys.path.append(os.getcwd()+'/experiment/src')
from Formation import FormationPlanner

if __name__ == "__main__":
    # initialize a planner with Motion Capture System for multiple agents
    MyPlanner = FormationPlanner()

    # run the planner online
    asyncio.run(MyPlanner.run_planner())
    asyncio.ensure_future(MyPlanner.run_planner())
