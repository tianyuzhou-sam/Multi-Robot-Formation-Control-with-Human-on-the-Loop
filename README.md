# Multi-Robot-Formation-Control-with-Human-on-the-Loop
Click for the video demonstration.

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/-3WkRDwjgGo/0.jpg)](https://youtu.be/-3WkRDwjgGo)

Dependencies
============
For this repo:
* Python >= 3.8
* [numpy](https://numpy.org/)
* [matplotlib](https://matplotlib.org/)
* [SciPy](https://www.scipy.org/)
* [CasADi](https://web.casadi.org/)

```
$ pip3 install numpy matplotlib scipy casadi
```
The experiment uses one [Clearpath Jackal](https://clearpathrobotics.com/jackal-small-unmanned-ground-vehicle/) robot and three Parrot Mambo quadrotors, the following dependancy is used:
* [Mambo-Tracking-Interface](https://github.com/tianyuzhou-sam/Mambo-Tracking-Interface)
For reference Jackal

Simulation
============
Make build directory
```
$ cd <MAIN_DIRECTORY>
$ mkdir build
```
Initialize the pre-generated waypoints and time indices, see test/test_onloopUI for details. 

To run the simulator:
```
$ python3 test/test_human_on_loop.py
```
To run the human interface:
```
$ python3 test/test_onLoopUI.py
```

Input the waypoints in the UI window and the time indices in the terminal window. If multiple waypoints are selected at the same time, separate the time indices by space (timeindex1 timeindex2 ...). The window will always open for additional waypoints.

![Alt text](/images/UI.png?raw=true "Optional Title")

An example of simulation is the following. The dashed red line indicates the original planned path, and the solid red line indicates the updated path after the guidance is given.

![Alt text](/images/3DFigure.png?raw=true "Optional Title")


Experiment
==========
Run Jackal and three Mambo with Qualisys Motion Capture System and Online Planner.
* Create a directory for csv trajectories
```
$ cd <Mambo-Tracking-Interface>/scripts_aimslab/
$ mkdir traj_csv_files
$ mkdir traj_csv_files/mambo_01
$ mkdir traj_csv_files/mambo_02
$ mkdir traj_csv_files/mambo_03
```
* Run Qualisys Motion Capture System
```
$ python3 scripts_aimslab/run_mocap_qualisys_for_formation.py 1
$ python3 scripts_aimslab/run_mocap_qualisys_for_formation.py 2
$ python3 scripts_aimslab/run_mocap_qualisys_for_formation.py 3
```
* Takeoff Mambo
```
$ python3 scripts_aimslab/run_mambo.py 1
$ python3 scripts_aimslab/run_mambo.py 2
$ python3 scripts_aimslab/run_mambo.py 3
```
* Run formation controller
```
$ cd <MAIN_DIRECTORY>
$ python3 experiment/run_Formation.py
```
* Run MPC controller (no human interface UI)
```
$ python3 experiment/run_MPC.py
```
* Run MPC controller and Human interaction UI On Jackal
```
$ python3 experiment/run_MPC_on_Loop.py
$ python3 test/test_onLoopUI.py
```
