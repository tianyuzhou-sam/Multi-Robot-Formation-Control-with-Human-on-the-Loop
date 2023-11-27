#!/usr/bin/env python3
import sys
import shutil
import subprocess
import casadi as ca
import numpy as np


class QuadSys:
    configDict: dict  # a dictionary for parameters
    dimStates: int  # dimension of states
    dimInputs: int  # dimension of inputs

    def __init__(self, configDict: dict, g:9.81, m:0, Ixx:0, Iyy:0, Izz:0):
        self.configDict = configDict
        self.dt = float(self.configDict["dt"])
        self.strLib = "QuadSys"
        self.prefixBuild = "build/"
        buildFlag = False

        self.dimStates = 12
        self.dimInputs = 4

        states = ca.SX.sym("x", self.dimStates)
        inputs = ca.SX.sym("u", self.dimInputs)

        # u = states[3]
        # v = states[4]
        # w = states[5]
        # pspi = states[6]
        # theta = states[7]
        # phi = states[8]
        # p = states[9]
        # q = states[10]
        # r = states[11]

        states = ca.SX.sym("x", self.dimStates)
        inputs = ca.SX.sym("u", self.dimInputs)

        # continuous-time dynamical function in casadi SX
        _contDyn = ca.SX.zeros(self.dimStates, 1)
        _contDyn[0, 0] = states[3]
        _contDyn[1, 0] = states[4]
        _contDyn[2, 0] = states[5]
        _contDyn[3, 0] = inputs[0]/m*(np.cos(states[6])*np.sin(states[7])*np.cos(states[8]) + np.sin(states[6])*np.sin(states[8]))
        _contDyn[4, 0] = inputs[0]/m*(np.cos(states[6])*np.sin(states[7])*np.sin(states[8]) - np.sin(states[6])*np.cos(states[8]))
        _contDyn[5, 0] = (inputs[0]/m*(np.cos(states[6])*np.cos(states[7])) - g)
        _contDyn[6, 0] = states[9]
        _contDyn[7, 0] = states[10]
        _contDyn[8, 0] = states[11]
        _contDyn[9, 0] = (Iyy-Izz)/Ixx*states[10]*states[11]+inputs[1]/Ixx
        _contDyn[10, 0] = (Izz-Ixx)/Iyy*states[11]*states[9]+inputs[2]/Iyy
        _contDyn[11, 0] = (Ixx-Iyy)/Izz*states[9]*states[10]+inputs[3]/Izz


        _discDyn = states + self.dt * _contDyn

        # Acceleration function
        prevInputs = ca.SX.sym("uPrev", self.dimInputs)
        _linearAcc = (inputs[0]-prevInputs[0]) / self.dt
        _angularAcc = (inputs[1]-prevInputs[1]) / self.dt

        # in casadi.Function
        self._contDynFun = ca.Function("contDynFun", [states, inputs], [_contDyn])
        self._discDynFun = ca.Function("discDynFun", [states, inputs], [_discDyn])
        self._linearAccFun = ca.Function("linearAccFun", [prevInputs, inputs], [_linearAcc])
        self._angularAccFun = ca.Function("angularAccFun", [prevInputs, inputs], [_angularAcc])

        # build casadi Function if True
        if buildFlag:
            self.build()

            # load Function from library
            libName = self.prefixBuild + self.strLib + ".so"
            self.contDynFun = ca.external("contDynFun", libName)
            self.discDynFun = ca.external("discDynFun", libName)

    def build(self):
        """
        Generate C codes and compile them.
        """
        # convert casadi.Function to C codes
        codeGen = ca.CodeGenerator(self.strLib + ".c")
        codeGen.add(self._contDynFun)
        codeGen.add(self._discDynFun)

        codeGen.generate(self.prefixBuild)

        # if OS is linux, compatible with Python <= 3.3
        if sys.platform == "linux" or sys.platform == "linux2":
            compiler = "gcc"
        # if OS is MacOS
        elif sys.platform == "darwin":
            compiler = "clang"
        else:
            raise Exception("Only supports Linux or MacOS!")

        # compile
        cmd_args = compiler + " -fPIC -shared -O3 " + self.prefixBuild + self.strLib + ".c -o " + self.prefixBuild + self.strLib + ".so"
        print("Run the following command:")
        print(cmd_args)
        subprocess.run(cmd_args, shell=True, check=True, capture_output=True)

    def forwardPropagate(self, initialState, uAll, timeStepArray, stepNumHorizon, tNow=0.0):
        """
        Forward propagate the dynamics given an initial condition and a sequence of inputs.

        Input:
            initialState: 1d numpy array, the initial state
            uAll: 1d numpy array, the sequence of inputs
                [u_0(0),u_1(0), u_0(1),u_1(1), ..., u_0(T-1),u_1(T-1)]
            timeStep: float, length of discrete time step

        Output:
            timeTraj: 1d numpy array, [0, timeStep, 2*timeStep, ..., timeHorizon]

            xTraj: 2d numpy array, each row is the state at a time step
                [[state_0(0), state_1(0), state_2(0), ...],
                [state_0(1), state_1(1), state_2(1), ...],
                ...
                [state_0(T), state_1(T), state_2(T), ...]]

            uTraj: 2d numpy array, each row is the input at a time step
                [[u_0(0), u_1(0), ...],
                [u_0(1), u_1(1), ...],
                ...
                [u_0(T-1), u_1(T-1), ...]]
        """
        timeTraj = np.zeros(stepNumHorizon+1)
        xTraj = np.zeros((stepNumHorizon+1, self.dimStates))
        uTraj = np.zeros((stepNumHorizon, self.dimInputs))

        xNow = initialState
        timeTraj[0] = tNow  # starting time [sec]
        xTraj[0, :] = np.array(initialState)  # initial state
        for idx in range(stepNumHorizon):
            uNow = uAll[self.dimInputs*idx : self.dimInputs*(idx+1)]
            xNext = self.discDynFun(xNow, uNow)
            timeTraj[idx+1] = timeTraj[idx] + timeStepArray[idx]  # time [sec]

            # casadi array to 1d numpy array
            xTraj[idx+1, :] = np.array(xNext).reshape((1,-1)).flatten()
            uTraj[idx, :] = np.array(uNow)  # input
            xNow = xNext
        return timeTraj, xTraj, uTraj

    def dir_cosine(self, q):
        C_B_I = ca.vertcat(
            ca.horzcat(1 - 2 * (q[2] ** 2 + q[3] ** 2), 2 * (q[1] * q[2] + q[0] * q[3]), 2 * (q[1] * q[3] - q[0] * q[2])),
            ca.horzcat(2 * (q[1] * q[2] - q[0] * q[3]), 1 - 2 * (q[1] ** 2 + q[3] ** 2), 2 * (q[2] * q[3] + q[0] * q[1])),
            ca.horzcat(2 * (q[1] * q[3] + q[0] * q[2]), 2 * (q[2] * q[3] - q[0] * q[1]), 1 - 2 * (q[1] ** 2 + q[2] ** 2))
        )
        return C_B_I
    
    def skew(self, v):
        v_cross = ca.vertcat(
            ca.horzcat(0, -v[2], v[1]),
            ca.horzcat(v[2], 0, -v[0]),
            ca.horzcat(-v[1], v[0], 0)
        )
        return v_cross
    
    def Omega(self, w):
        omeg = ca.vertcat(
            ca.horzcat(0, -w[0], -w[1], -w[2]),
            ca.horzcat(w[0], 0, w[2], -w[1]),
            ca.horzcat(w[1], -w[2], 0, w[0]),
            ca.horzcat(w[2], w[1], -w[0], 0)
        )
        return omeg
    
    def quaternion_mul(self, p, q):
        return ca.vertcat(p[0] * q[0] - p[1] * q[1] - p[2] * q[2] - p[3] * q[3],
                       p[0] * q[1] + p[1] * q[0] + p[2] * q[3] - p[3] * q[2],
                       p[0] * q[2] - p[1] * q[3] + p[2] * q[0] + p[3] * q[1],
                       p[0] * q[3] + p[1] * q[2] - p[2] * q[1] + p[3] * q[0])


if __name__ == '__main__':
    # dictionary for configuration
    # dt for Euler integration
    configDict = {"dt": 0.1}

    buildFlag = True
    # buildFlag = False

    # initialize Quadrotor System
    MyQuad = QuadSys(configDict=configDict, buildFlag=buildFlag)

    x0 = np.array([0, 0, 0, 0, 0, 0, 0, 0 ,0, 0, 0, 0])
    u0= np.array([1.0, 1.0, 1.0, 1.0])

    x0_dot = MyQuad._contDynFun(x0, u0)
    x1 = MyQuad._discDynFun(x0, u0)
    print("x0_dot: ", x0_dot)
    print("x1: ", x1)


