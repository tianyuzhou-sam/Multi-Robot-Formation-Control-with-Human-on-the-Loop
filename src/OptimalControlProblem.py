#!/usr/bin/env python3
import sys
import shutil
import subprocess
import copy
import time
import casadi as ca
import numpy as np
from JackalSys import JackalSys


class OptimalControlProblem:
    configDict: dict  # a dictionary for parameters

    def __init__(self, configDict: dict, MyJackalSys: JackalSys, buildFlag=True, target=[0,0]):
        self.configDict = configDict
        self.MyJackalSys = MyJackalSys
        self.target = target
        self.prefixBuild = "build/"
        self.strSolver = "nlp_uniform"
        self.strLib = "lib_oc_uniform"


        self.w1 = 1.0
        self.w2 = 0.1
        self.w3 = 0.1

        try:
            self.w1 = float(self.configDict["weights"][0])
            self.w2 = float(self.configDict["weights"][1])
            self.w3 = float(self.configDict["weights"][2])
        except:
            print("No setting for weights. Set w1 - w3 by default.")


        self.lin_vel_lb = 0.0
        self.lin_vel_ub = 0.25
        self.ang_vel_lb = -0.5
        self.ang_vel_ub = 0.5
        self.lin_acc_lb = -0.1
        self.lin_acc_ub = 0.1
        self.ang_acc_lb = -0.1 
        self.ang_acc_ub = 0.1

        # casadi nlp options
        self.optsIpopt = {"print_level": 5, "sb": "yes"}
        self.optsCasadi = {"print_time": False, "ipopt": self.optsIpopt}

        assert(self.MyJackalSys.dt == self.configDict["dt"])
        self.dt = float(self.configDict["dt"])  # second
        self.stepNumHorizon = int(self.configDict["stepNumHorizon"])
        self.timeHorizon = self.dt * self.stepNumHorizon  # second

        self.dimStates = self.MyJackalSys.dimStates
        self.dimInputs = self.MyJackalSys.dimInputs

        # decision variable is the column stack of all states (excluding initial state) and all inputs
        self.dimDecision = self.dimStates * self.stepNumHorizon + \
            self.dimInputs * self.stepNumHorizon

        # time trajectory for optimal control
        self.timeTraj = self.dt * np.linspace(0, self.stepNumHorizon, num=self.stepNumHorizon+1)

        # constraints, dynamics equality constaints + other constraints
        self.dimConstraint = self.stepNumHorizon*self.dimStates + 2*(self.stepNumHorizon-1)

        # build the NLP object if True
        if buildFlag:
            self.buildOptimalControl()

        # create a new NLP solver instance from the compiled code
        self.solver = ca.nlpsol("solver", "ipopt", self.prefixBuild + self.strSolver + ".so", self.optsCasadi)

        # load Function from library
        libName = self.prefixBuild + self.strLib + ".so"
        self.dynamicCstrFun = ca.external("dynamicCstrFun", libName)

        # construct bounds
        self.constructBounds()

        # construct computing starting point method
        # use zero inputs
        if self.configDict["startPointMethod"] == "zeroInput":
            self.computeStartingPoint = lambda iniState: self._computeStartingPoint(iniState)
        # use basic PID control
        elif self.configDict["startPointMethod"] == "PID":
            self.computeStartingPoint = lambda iniState: self._computeStartingPointPID(iniState)
        else:
            print("No setting for computing starting point. Set to zeroInput by default.")
            self.computeStartingPoint = lambda iniState: self._computeStartingPoint(iniState)

    def solve(self, iniState: np.array, timeNow=0.0):
        """
        Solve one instance of optimal control.

        Return:
            xTraj: 2d numpy array, each row is the state at a time step, including the initial state
                [[state_0(0), state_1(0), state_2(0), ...],
                [state_0(1), state_1(1), state_2(1), ...],
                ...
                [state_0(T), state_1(T), state_2(T), ...]]

            uTraj: 2d numpy array, each row is the input at a time step
                [[u_0(0), u_1(0), ...],
                [u_0(1), u_1(1), ...],
                ...
                [u_0(T-1), u_1(T-1), ...]]
            
            timeTraj: 1d numpy array, from initial time to final time
        """
        # compute initial point for all decision variables given the initial state with zero inputs
        decisionIni = self.computeStartingPoint(iniState)
        # revise arguments for the solver
        self.args["p"] = ca.DM(iniState)
        self.args["x0"] = ca.DM(decisionIni)

        # solve
        t0 = time.time()
        res = self.solver(**self.args)
        t1 = time.time()
        ipoptTime = t1 - t0

        # status
        returnStatus = self.solver.stats()["return_status"]
        successFlag = self.solver.stats()["success"]

        # # Print solution
        # print("-----")
        # print("objective at solution =", res["f"])
        # print("primal solution =", res["x"])
        # print("dual solution (x) =", res["lam_x"])
        # print("dual solution (g) =", res["lam_g"])

        xTraj = np.reshape(res["x"][0:self.dimStates*self.stepNumHorizon], (self.stepNumHorizon, self.dimStates))
        xTraj = np.vstack((iniState, xTraj))
        uTraj = np.reshape(res["x"][self.dimStates*self.stepNumHorizon:self.dimDecision], (self.stepNumHorizon, self.dimInputs))
        timeTraj = self.timeTraj + timeNow

        return xTraj, uTraj, timeTraj, ipoptTime, returnStatus, successFlag

    def buildOptimalControl(self):
        # symbolic variable for decision variable
        decisionAllSym = ca.SX.sym("z", self.dimDecision)

        # symbolic variable for initial state, will be a parameter of NLP
        iniStateSym = ca.SX.sym("x0", self.dimStates)

        # create cost function
        _costFun = self._costFun(decisionAllSym, iniStateSym)

        # create constraints
        _cstrFun = ca.SX.zeros(self.dimConstraint)
        _cstrFun[0 : self.stepNumHorizon * self.dimStates] = self._dynamicCstrFun(decisionAllSym, iniStateSym)
        _cstrFun[self.stepNumHorizon * self.dimStates : self.dimConstraint] = self._otherCstrFun(decisionAllSym, iniStateSym)

        # create an NLP problem structure
        nlp = {"x": decisionAllSym, "f": _costFun, "g": _cstrFun, "p": iniStateSym}

        # create an NLP solver instance
        solver = ca.nlpsol("solver", "ipopt", nlp)

        # generate C code for the NLP functions
        name_nlp = self.strSolver + ".c"
        solver.generate_dependencies(name_nlp)

        # move source code to build/
        shutil.move(name_nlp, self.prefixBuild + name_nlp)

        # if OS is linux, compatible with Python <= 3.3
        if sys.platform == "linux" or sys.platform == "linux2":
            compiler = "gcc"
        # if OS is MacOS
        elif sys.platform == "darwin":
            compiler = "clang"
        else:
            raise Exception("Only supports Linux or MacOS!")

        # compile
        cmd_args = compiler + " -fPIC -shared -O3 " + self.prefixBuild + name_nlp + " -o " + self.prefixBuild + self.strSolver + ".so"
        print("Run the following command:")
        print(cmd_args)
        subprocess.run(cmd_args, shell=True, check=True, capture_output=True)

        # convert casadi.Function to C codes
        codeGen = ca.CodeGenerator(self.strLib + ".c")
        __dynamicCstrFun = ca.Function("dynamicCstrFun", [decisionAllSym, iniStateSym], [self._dynamicCstrFun(decisionAllSym, iniStateSym)])
        codeGen.add(__dynamicCstrFun)
        codeGen.generate(self.prefixBuild)

        # compile
        cmd_args = compiler + " -fPIC -shared -O3 " + self.prefixBuild + self.strLib + ".c -o " + self.prefixBuild + self.strLib + ".so"
        print(cmd_args)
        subprocess.run(cmd_args, shell=True, check=True, capture_output=True)

    def constructBounds(self):
        # define bounds for decision variables
        stateAllLb = ca.DM([-ca.inf, -ca.inf, -ca.inf])
        stateAllLb = ca.repmat(stateAllLb, self.stepNumHorizon)

        stateAllUb = ca.DM([ca.inf, ca.inf, ca.inf])
        stateAllUb = ca.repmat(stateAllUb, self.stepNumHorizon)

        inputAllLb = ca.DM([self.lin_vel_lb, self.ang_vel_lb])
        inputAllLb = ca.repmat(inputAllLb, self.stepNumHorizon)

        inputAllUb = ca.DM([self.lin_vel_ub, self.ang_vel_ub])
        inputAllUb = ca.repmat(inputAllUb, self.stepNumHorizon)

        self.decisionLb = ca.vertcat(stateAllLb, inputAllLb)
        self.decisionUb = ca.vertcat(stateAllUb, inputAllUb)

        # define bounds for constraints
        dynCstrLb = ca.DM.zeros(self.stepNumHorizon*self.dimStates)
        dynCstrUb = ca.DM.zeros(self.stepNumHorizon*self.dimStates)

        otherCstrLb = ca.DM([self.lin_acc_lb, self.ang_acc_lb])
        otherCstrLb = ca.repmat(otherCstrLb, self.stepNumHorizon-1)

        otherCstrUb = ca.DM([self.lin_acc_ub, self.ang_acc_ub])
        otherCstrUb = ca.repmat(otherCstrUb, self.stepNumHorizon-1)

        self.cstrLb = ca.vertcat(dynCstrLb, otherCstrLb)
        self.cstrUb = ca.vertcat(dynCstrUb, otherCstrUb)

        # arguments for solving one instance of optimal control
        # these two are just for initialization
        x0 = ca.DM.zeros(self.dimStates)
        decisionIni = ca.DM.zeros(self.dimDecision)
        self.args = {"lbx": self.decisionLb, "ubx": self.decisionUb, "lbg": self.cstrLb, "ubg": self.cstrUb, "x0": decisionIni, "p": x0}


    def _costFun(self, decisionAll, iniState):
        xAll = decisionAll[0 : self.dimStates * self.stepNumHorizon]
        uAll = decisionAll[self.dimStates * self.stepNumHorizon : self.dimDecision]
        cost = 0.0

        # for soc
        for idx in range(self.stepNumHorizon):
            if idx == 0:
                xNow = iniState
            else:
                xNow = xAll[self.dimStates*(idx-1) : self.dimStates*idx]
            cost += self.w1 * ((xNow[0] - self.target[0]) **2 + (xNow[1] - self.target[1]) **2)

        return cost

    def _dynamicCstrFun(self, decisionAll, iniState):
        """
        Equality constraints for dynamical system.

        0 <= fun <= 0
        """
        xAll = decisionAll[0 : self.dimStates * self.stepNumHorizon]
        uAll = decisionAll[self.dimStates * self.stepNumHorizon : self.dimDecision]
        fun = ca.SX.zeros(self.stepNumHorizon * self.dimStates)

        for idx in range(self.stepNumHorizon):
            if idx == 0:
                xNow = iniState
            else:
                xNow = xAll[self.dimStates*(idx-1) : self.dimStates*idx]
            xNext = xAll[self.dimStates*idx : self.dimStates*(idx+1)]
            uNow = uAll[self.dimInputs*idx : self.dimInputs*(idx+1)]
            fun[self.dimStates*idx : self.dimStates*(idx+1)] = xNext - self.MyJackalSys._discDynFun(xNow, uNow)

        return fun

    def _otherCstrFun(self, decisionAll, iniState):
        """
        Constraints that are included here:
        lin_acc_lb <= lin_acc <= lin_acc_ub
        ang_acc_lb <= ang_acc <= ang_acc_ub
        """
        xAll = decisionAll[0 : self.dimStates * self.stepNumHorizon]
        uAll = decisionAll[self.dimStates * self.stepNumHorizon : self.dimDecision]
        fun = ca.SX.zeros(2 * (self.stepNumHorizon - 1))

        uPrev = uAll[0: self.dimInputs]

        for idx in range(self.stepNumHorizon-1):
            uNow = uAll[self.dimInputs*(idx+1) : self.dimInputs*(idx+2)]
            fun[2*idx] = (uNow[0] - uPrev[0])/self.dt
            fun[2*idx+1] = (uNow[1] - uPrev[1])/self.dt
            uPrev = uNow

        fun_total = fun

        # assert(fun_total.shape[0] == 3*self.stepNumHorizon)
        return fun_total


    def _computeStartingPoint(self, initialState: np.array):
        """
        Compute the starting point of the optimization problem by forward propagating the dynamics
        given an initial condition and zeros inputs.

        Input:
            initialState: 1d numpy array, the initial state

        Output:
            decisionAll: 1d numpy array, a column stack of all states and inputs, excluding the initial condition
                [state(1), ..., state(T), input(0), ..., input(T-1)]
        """
        # uAll: 1d numpy array, the sequence of inputs,
        # [u(0), u(1), ..., u(T-1)]
        # generate random inputs for testing only
        # uAll = np.random.uniform(-5, 5, self.DynSystem.dimInputsAll)

        uAll = np.zeros(self.dimInputs * self.stepNumHorizon)
        xAll = np.zeros(self.dimStates * self.stepNumHorizon)

        xNow = initialState
        for idx in range(self.stepNumHorizon):
            xNext = self.MyJackalSys.discDynFun(xNow, uAll[self.dimInputs*idx : self.dimInputs*(idx+1)])
            xAll[idx*self.dimStates : (idx+1)*self.dimStates] = np.array(xNext).flatten()
            xNow = xNext
        decisionAll = np.concatenate((xAll, uAll))

        # re = self.dynamicCstrFun(decisionAll, initialState)
        # norm_check = np.linalg.norm(np.array(re).flatten())
        # print("norm_check: ", norm_check)
        # if abs(norm_check) > 0.01:
        #     print("norm_check: ", norm_check)
        #     raise Exception("starting point equality constraint check (should be zero)")

        return decisionAll



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
    MyOC = OptimalControlProblem(configDict, MyJackal, buildFlag, target)

    # test
    x0 = np.array([0, 0, 0])
    u0 = np.array([0, 0])
    decisionAll = MyOC._computeStartingPoint(x0)
    print(decisionAll)

    # solve
    xTraj, uTraj, timeTraj, ipoptTime, returnStatus, successFlag = MyOC.solve(x0, timeNow=0.0)
    print(xTraj)
    print(uTraj)
