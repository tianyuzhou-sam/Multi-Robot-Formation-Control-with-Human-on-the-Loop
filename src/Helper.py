#!/usr/bin/env python3
import copy
import numpy as np
import scipy.io as sio
import scipy
from ModelPredictiveControl import ModelPredictiveControl
# from ModelPredictiveControlEKF import ModelPredictiveControlEKF
from statsmodels.stats.multicomp import pairwise_tukeyhsd


def evaluateData(caseNum, matFilePrefix, suffix, plotFlag=True, logPrintFlag=False):
    # data file name
    fileName = matFilePrefix + suffix
    # load data
    result = sio.loadmat(fileName)["data"]
    # dictionary for configuration, not important
    configDict = {
        "dt": 5.0, "stepNumHorizon": 40,
        "T_amb": 25+273.15, "startPointMethod": "zeroInput",
        "weights": [40.0, 0.1, 0.1], "method": "MPC",
        "CRate": 1.0, "TcoreDes": 25}
    if caseNum == 4:
        configDict["PactBound"] = [-24.0, 24.0]
        configDict["weights"] = [40.0, 0.1, 0.1, 0.1]
        configDict["trackTcoreFlag"] = True
    else:
        configDict["PactBound"] = [-8.0, 8.0]

    buildFlag = False
    # initialize ModelPredictiveControl
    MyMPC = ModelPredictiveControl(configDict, buildFlag)
    if plotFlag:
        # visualization
        MyMPC.visualize(result, matFlag=True, legendFlag=False, titleFlag=False, blockFlag=False)
    # print results
    ipoptTimeTraj = printResult(result, matFlag=True, logPrintFlag=logPrintFlag, allPrintFlag=False)
    return ipoptTimeTraj

def evaluateDataEKF(matFilePrefix, suffix, plotFlag=True, logPrintFlag=False, plotBeforeFlag=True):
    # data file name
    fileName = matFilePrefix + suffix
    # load data
    result = sio.loadmat(fileName)["data"]
    # dictionary for configuration, not important
    configDict = {
        "dt": 5.0, "stepNumHorizon": 40,
        "T_amb": 25+273.15, "startPointMethod": "zeroInput",
        "weights": [40.0, 0.1, 0.1], "method": "MPC",
        "CRate": 1.0, "TcoreDes": 25}
    buildFlag = False
    # initialize ModelPredictiveControlEKF
    MyMPC = ModelPredictiveControlEKF(configDict, buildFlag)
    if plotFlag:
        # visualization
        MyMPC.visualize(result, matFlag=True, plotBeforeFlag=plotBeforeFlag, legendFlag=False, titleFlag=False, blockFlag=False)
    # print results
    ipoptTimeTraj = printResult(result, matFlag=True, logPrintFlag=logPrintFlag, allPrintFlag=False)
    return ipoptTimeTraj

def runSimTempCtrlPID(TcoreDes, configDictInput, matFilePrefix, iniState, timeTotal, buildFlag, saveFlag, plotFlag):
    """
    Input:
        TcoreDes: float, desired core temperature in Celsius
    """
    ################# 5-th method, with external heating and cooling, separate temperature PID control
    configDict = copy.deepcopy(configDictInput)
    configDict["method"] = "tempCtrlPID"
    configDict["TcoreDes"] = TcoreDes + 273.15
    configDict["kp"] = 0.5
    configDict["kd"] = 150.0
    configDict["ki"] = 0.010
    # configDict["kp"] = 1.8
    # configDict["kd"] = 0.0
    # configDict["ki"] = 0.0
    # initialize ModelPredictiveControl
    MyMPC = ModelPredictiveControl(configDict, buildFlag)
    # run simulation
    result = MyMPC.run(iniState, timeTotal)
    # evaluate the trajectory
    result = MyMPC.evaluateTraj(result)
    # save the results∂
    if saveFlag:
        sio.savemat(matFilePrefix+"_tempCtrlPID_"+str(int(TcoreDes))+".mat", {'data': result})
    if plotFlag:
        # visualization
        MyMPC.visualize(result, matFlag=False, legendFlag=True, titleFlag=False, blockFlag=False)
    del MyMPC, configDict
    return result


def printResult(result: dict, matFlag=False, logPrintFlag=False, allPrintFlag=True):
    """
    Print results

    If result is loaded from a .mat file, matFlag = True
    If result is from a seld-defined dict variable, matFlag = False
    """
    algTimeTraj = result["algTimeTraj"]
    ipoptTimeTraj = result["ipoptTimeTraj"]
    timeCharge = result["timeCharge"]
    finishFlag = result["finishFlag"]
    socTraj = result["socTraj"]
    xTraj = result["xTraj"]
    energyJoule = result["energyJoule"]
    energyChargeNeat = result["energyChargeNeat"]
    energyCharge = result["energyCharge"]
    energyHeatCool = result["energyHeatCool"]
    energyTotal = result["energyTotal"]
    logTimeTraj = result["logTimeTraj"]
    logStrTraj = result["logStrTraj"]
    if matFlag:
        algTimeTraj = algTimeTraj[0, 0].flatten()
        ipoptTimeTraj = ipoptTimeTraj[0, 0].flatten()
        timeCharge = timeCharge[0, 0][0, 0]
        finishFlag = finishFlag[0, 0][0, 0]
        socTraj = socTraj[0, 0].flatten()
        xTraj = xTraj[0, 0]
        energyJoule = energyJoule[0, 0][0, 0]
        energyChargeNeat = energyChargeNeat[0, 0][0, 0]
        energyCharge = energyCharge[0, 0][0, 0]
        energyHeatCool = energyHeatCool[0, 0][0, 0]
        energyTotal = energyTotal[0, 0][0, 0]
        logTimeTraj = logTimeTraj[0, 0].flatten()
        logStrTraj = logStrTraj[0, 0].flatten()

    # computing time [sec]
    time_mean = np.mean(ipoptTimeTraj)
    time_std = np.std(ipoptTimeTraj)
    print("Average computing time [ms]: ", round(1000*time_mean, 2))
    print("Std computing time [ms]: ", round(1000*time_std, 2))
    print("Charging time [sec]: ", timeCharge)
    print("Whether fully charged: ", bool(finishFlag))

    try:
        xEstTraj = result["xEstTraj"]
        errSocTraj = result["errSocTraj"]
        errVoltageTraj = result["errVoltageTraj"]
        if matFlag:
            xEstTraj = xEstTraj[0, 0]
            errSocTraj = errSocTraj[0, 0].flatten()
            errVoltageTraj = errVoltageTraj[0, 0].flatten()
        
        errSocPerTraj = errSocTraj / socTraj * 100
    
        errStatePerTraj = abs(xEstTraj - xTraj) / xTraj * 100
        errorStatePerMax = errStatePerTraj.max(axis=0)
        errorStatePerMean = np.mean(errStatePerTraj, axis=0)
        errorStatePerStd = np.std(errStatePerTraj, axis=0)
        errorStatePerMedian = np.median(errStatePerTraj, axis=0)
        errorStatePer25 = np.percentile(errStatePerTraj, 25, axis=0)
        errorStatePer75 = np.percentile(errStatePerTraj, 75, axis=0)

        print("EKF Vb error [%] max: ", round(errorStatePerMax[0],2))
        print("EKF Vb error [%] mean ± std: ", str(round(errorStatePerMean[0],2))+" ± "+str(round(errorStatePerStd[0],2)))
        print("EKF Vb error [%] [25th, median, 75th percentile]: ", str(round(errorStatePer25[0],2))+", "+\
            str(round(errorStatePerMedian[0],2))+", "+str(round(errorStatePer75[0],2)))

        print("EKF Vs error [%] max: ", round(errorStatePerMax[1],2))
        print("EKF Vs error [%] mean ± std: ", str(round(errorStatePerMean[1],2))+" ± "+str(round(errorStatePerStd[1],2)))
        print("EKF Vs error [%] [25th, median, 75th percentile]: ", str(round(errorStatePer25[1],2))+", "+\
            str(round(errorStatePerMedian[1],2))+", "+str(round(errorStatePer75[1],2)))

        print("EKF Tcore error [%] max: ", round(errorStatePerMax[2],2))
        print("EKF Tcore error [%] mean ± std: ", str(round(errorStatePerMean[2],2))+" ± "+str(round(errorStatePerStd[2],2)))
        print("EKF Tcore error [%] [25th, median, 75th percentile]: ", str(round(errorStatePer25[2],2))+", "+\
            str(round(errorStatePerMedian[2],2))+", "+str(round(errorStatePer75[2],2)))

        print("EKF Tsurf error [%] max: ", round(errorStatePerMax[3],2))
        print("EKF Tsurf error [%] mean ± std: ", str(round(errorStatePerMean[3],2))+" ± "+str(round(errorStatePerStd[3],2)))
        print("EKF Tsurf error [%] [25th, median, 75th percentile]: ", str(round(errorStatePer25[3],2))+", "+\
            str(round(errorStatePerMedian[3],2))+", "+str(round(errorStatePer75[3],2)))

        print("EKF soc error [%] max: ", round(errSocPerTraj.max(),2))
        print("EKF soc error [%] mean ± std: ", str(round(np.mean(errSocPerTraj),2))+" ± "+str(round(np.std(errSocPerTraj),2)))
        print("EKF soc error [%] [25th, median, 75th percentile]: ", str(round(np.percentile(errSocPerTraj,25),2))+", "+\
            str(round(np.median(errSocPerTraj),2))+", "+str(round(np.percentile(errSocPerTraj,75),2)))
    except:
        pass

    if allPrintFlag:
        print("Joule loss energy [kJ]: ", round(energyJoule/1000, 2))
        print("Neat energy for charging [kJ]: ", round(energyChargeNeat/1000, 2))
        print("Energy for charging + Joule loss [kJ]: ", round(energyCharge/1000, 2))
        print("Joule loss percentage [%]: ", round(100*energyJoule/energyCharge, 2))
        print("Energy for heating and cooling [kJ]: ", round(energyHeatCool/1000, 2))
    print("Total energy [kJ]: ", round((energyCharge+energyHeatCool)/1000, 2))
    print("Total efficiency [%]: ", round(100*energyChargeNeat/(energyCharge+energyHeatCool), 2))
    if logPrintFlag:
        if (logTimeTraj.size != 0) or (logStrTraj.size != 0):
            print("Unsuccessful IPOPT time:")
            print(logTimeTraj)
            print("Unsuccessful IPOPT reason:")
            print(logStrTraj)
    print("")
    return ipoptTimeTraj

def oneWayANOVA(ipoptTimeTrajList, groupStrList):
    """
    Input:
        ipoptTimeTrajList: 1d list of 1d numpy array, each element is a 1d numpy array
        groupStrList: 1d list of string, each element is the name
    
    Output:
        F_OnewayResult: tuple with 2 elements, F statistic and p value
    """

    resultANOVA = scipy.stats.f_oneway(*ipoptTimeTrajList)
    print(resultANOVA)

    # Tukey's Test
    if resultANOVA[1] <= 0.05:
        dataAll = list()
        groupAll = list()
        for idx in range(len(ipoptTimeTrajList)):
            group = np.repeat([groupStrList[idx]], repeats=ipoptTimeTrajList[idx].size).tolist()
            data = ipoptTimeTrajList[idx].tolist()
            groupAll.extend(group)
            dataAll.extend(data)
        tukey = pairwise_tukeyhsd(endog=dataAll, groups=groupAll, alpha=0.05)
        print(tukey)


if __name__ == '__main__':
    a = np.array([1, 2, 3, 4])
    b = np.array([4, 5, 6, 7])
    c = np.array([-1, -2, -3, -4])
    aaa = [a, b, c]
    oneWayANOVA(aaa)

    # If p value < 0.05, we can reject the null hypothesis.
    # This implies that we have sufficient proof to say that
    # there exists a difference in the data among 3 groups.
