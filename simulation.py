import csv
import os
import pickle
import shutil
import time
import math

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import functions
import main
from parameters import par

matplotlib.use('TkAgg')

"""Simulation Options"""
beliefOptions = ['seqBayes']  # 'stkf' 'seqBayes', 'regBayes', 'regBayesTrunc'
controlOptions = ['cbts', 'pi2']  # 'cbts', 'pi2', 'randomWalk', 'geist'
cbtsNodeBelief = ['noUpdates']  # 'fullGMRF', 'sampledGMRF', 'noUpdates'

"""Simulation Options"""
printTime = False
saveToFile = True
nSim = 20
nIter = 500
fieldType = 'random'  # 'peak','sine', 'random' or 'predefined'
temporal = False  # True: time varying field
varTimeKernel = False
obstacle = True
plot = True
saveBeliefHistory = False

"PI2"
K = [15]
H = [10]
nUpdated = [10]
lambd = [0.1]
pi2ControlCost = [10]

"CBTS"
branchingFactor = [6]
maxDepth = [2]
kappa = [1]
kappaChildSelection = [1]
UCBRewardFactor = [0]
cbtsControlCost = [0.1]
discountFactor = [0.8]

"Initialize lists and dicts"
simCaseList = []
parSettingsList = []
wrmseDict = {}
rmseDict = {}
wTotalVarDict = {}
totalVarDict = {}

"""Iterate through simulation options and create parameter objects"""
for belief in beliefOptions:
    par.belief = belief

    for control in controlOptions:
        if control == 'pi2':
            for K_i in K:
                for H_i in H:
                    for nUpdated_i in nUpdated:
                        for lambd_i in lambd:
                            for pi2ControlCost_i in pi2ControlCost:
                                parObject = par(belief, control, 0, fieldType, temporal, varTimeKernel, obstacle,
                                plot, nIter, K_i, H_i, nUpdated_i, lambd_i, pi2ControlCost_i, 0, 0, 0, 0, 0, 0, 0)
                                parSettingsList.append(parObject)
                                simCaseList.append(belief + '_' + control + '_' + 'temporal' + str(temporal)
                                                   + '_' + 'varTimeKernel' + str(varTimeKernel)
                                                   + '_' + 'K' + str(K_i).replace('.', 'p')
                                                   + '_' + 'H' + str(H_i).replace('.', 'p')
                                                   + '_' + 'nUpdated' + str(nUpdated_i).replace('.', 'p')
                                                   + '_' + 'lambd' + str(lambd_i).replace('.', 'p')
                                                   + '_' + 'pi2ControlCost' + str(pi2ControlCost_i).replace('.', 'p'))
        elif control == 'cbts':
            for cbtsNodeBelief_i in cbtsNodeBelief:
                for branchingFactor_i in branchingFactor:
                    for maxDepth_i in maxDepth:
                        for kappa_i in kappa:
                            for kappaChildSelection_i in kappaChildSelection:
                                for UCBRewardFactor_i in UCBRewardFactor:
                                    for cbtsControlCost_i in cbtsControlCost:
                                        for discountFactor_i in discountFactor:
                                            parObject = par(belief, control, cbtsNodeBelief_i, fieldType, temporal,
                                                            varTimeKernel, obstacle, plot, nIter, 0, 0, 0, 0, 0,
                                                            branchingFactor_i, maxDepth_i, kappa_i,
                                                            kappaChildSelection_i, UCBRewardFactor_i, cbtsControlCost_i,
                                                            discountFactor_i)
                                            parSettingsList.append(parObject)
                                            simCaseList.append(belief + '_' + control + '_' + 'temporal' + str(temporal)
                                            + '_' + 'varTimeKernel' + str(varTimeKernel)
                                            + '_' + 'cbtsNodeBelief' + str(cbtsNodeBelief_i).replace('.', 'p')
                                            + '_' + 'branchingFactor' + str(branchingFactor_i).replace('.', 'p')
                                            + '_' + 'maxDepth' + str(maxDepth_i).replace('.', 'p')
                                            + '_' + 'kappa' + str(kappa_i).replace('.', 'p')
                                            + '_' + 'kappaChildSelection' + str(kappaChildSelection_i).replace('.', 'p')
                                            + '_' + 'UCBRewardFactor' + str(UCBRewardFactor_i).replace('.', 'p')
                                            + '_' + 'cbtsControlCost' + str(cbtsControlCost_i).replace('.', 'p')
                                            + '_' + 'discountFactor' + str(discountFactor_i).replace('.', 'p'))
        if control == 'randomWalk':
            parObject = par(belief, control, 0, fieldType, temporal, varTimeKernel, obstacle, plot, nIter, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0)
            parSettingsList.append(parObject)
            simCaseList.append(belief + '_' + control)

        if control == 'geist':
            parObject = par(belief, control, 0, fieldType, temporal, varTimeKernel, obstacle, plot, nIter, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0)
            parSettingsList.append(parObject)
            simCaseList.append(belief + '_' + control)

"""Create directory if data should be saved"""
if saveToFile:
    index = 0
    dirpath = os.getcwd()
    try:
        comment = input('Enter a comment for the simulation:')
    except:
        comment = ""
    folderName = time.strftime("%Y%m%d") + "_" + str(index) + comment
    path = dirpath + "/sim/" + folderName
    while os.path.exists(path):
        index += 1
        folderName = time.strftime("%Y%m%d") + "_" + str(index) + comment
        path = dirpath + "/sim/" + folderName
    try:
        os.mkdir(path)
    except:
        print("Error while creating directory!")
    os.chdir(path)

    # Copy used parameters
    shutil.copyfile(dirpath + '/parameters.py', path + '/' + folderName + '_parameters.txt')
    shutil.copyfile(dirpath + '/Config.py', path + '/' + folderName + '_config.txt')

for i in range(len(parSettingsList)):
    par = parSettingsList[i]
    simCase = simCaseList[i]

    """Initialize"""
    x = []
    y = []
    trueField = []
    gmrf = []
    gmrfMean = []
    gmrfDiagCov = []
    controller = []
    CBTS = []
    timeVec = []
    xHist = []
    yHist = []
    wrmse = []
    rmse = []
    wTotalVar = []
    totalVar = []

    for j in range(nSim):
        print("Simulation ", j+1, " of ", nSim, " with ", simCase)
        xR, yR, trueFieldR, gmrfR, controllerR, CBTSR, timeVecR, xHistR, yHistR, wrmseR, rmseR, wTotalVarR, totalVarR = \
            main.main(par, printTime, saveBeliefHistory, simCase)

        x.append(xR)
        y.append(yR)
        trueField.append(trueFieldR)
        gmrf.append(gmrfR)
        gmrfMean.append(gmrfR.meanCond)
        gmrfDiagCov.append(gmrfR.diagCovCond)
        controller.append(controllerR)
        CBTS.append(CBTSR)
        timeVec.append(timeVecR)
        xHist.append(xHistR)
        yHist.append(yHistR)
        wrmse.append(wrmseR)
        rmse.append(rmseR)
        wTotalVar.append(wTotalVarR)
        totalVar.append(totalVarR)

        "plot total and average calculation time:"
        print("\nTotal computation time is", np.sum(timeVecR), "s")
        print("Average computation time is", np.mean(timeVecR), "s")

        # Save gmrf for each simulation (since it can be very large)
        with open('objs_' + str(j) + '_gmrf_' + simCase + '.pkl', 'wb') as f:
            pickle.dump(gmrfR, f)

    wrmseDict[simCase] = wrmse
    rmseDict[simCase] = rmse
    wTotalVarDict[simCase] = wTotalVar
    totalVarDict[simCase] = totalVar

    if saveToFile:
        """Save objects"""
        # Save objects
        with open('objs_other_' + simCase + '.pkl', 'wb') as f:
            pickle.dump([x, y, trueField, controller, CBTS, timeVec, xHist, yHist,  wrmseR, rmseR, wTotalVarR, totalVarR], f)

        # Getting back the objects:
        # with open('objs.pkl','rb') as f:
        #   x, y, trueField, controller, CBTS, timeVec, xHist, yHist,  wrmseR, rmseR, wTotalVarR, totalVarR, simCase = pickle.load(f)

        # Save data as csv
        with open(folderName + '_' + simCase + '_data.csv', 'w') as dataFile:
            writer = csv.writer(dataFile)
            for k in range(nSim):
                writer.writerow([k])
                writer.writerow(x[k])
                writer.writerow(y[k])
                writer.writerow(timeVec[k])
                writer.writerow(xHist[k])
                writer.writerow(yHist[k])
                writer.writerow(wrmse[k])
                writer.writerow(rmse[k])
                writer.writerow(wTotalVar[k])
                writer.writerow(totalVar[k])
                writer.writerow(gmrfMean[k])
                writer.writerow(gmrfDiagCov[k])
                writer.writerow(["-"])
            dataFile.close()

    """Plot fields"""
    fig0 = plt.figure(100, figsize=(19.2, 10.8), dpi=100)
    print("Plotting..")
    for i in range(nSim):
        functions.plotFields(par, fig0, x[i], y[i], trueField[i], gmrf[i], controller[i], CBTS[i],
                             timeVec[i], xHist[i], yHist[i])
        if saveToFile:
            fig0.savefig(folderName + '_' + str(i) + '_' + simCase + '_' + par.fieldType + '.svg',
                         format='svg')
        else:
            plt.show()
        plt.clf()
    plt.close(fig0)

"""Plot Weighted Median and IQR"""
fig1 = plt.figure(201, figsize=(19.2, 10.8), dpi=100)
plt.title("Weighted Median and IQR")
functions.plotOverallPerformance(nIter, simCaseList, wrmseDict, wTotalVarDict, True, 'median')
if saveToFile:
    fig1.savefig(folderName + '_weightedMedianIQR.svg', format='svg')
else:
    plt.show()

"""Plot Weighted Mean and standard deviation"""
fig2 = plt.figure(202, figsize=(19.2, 10.8), dpi=100)
plt.title("Weighted Mean and Standard Deviation")
functions.plotOverallPerformance(nIter, simCaseList, wrmseDict, wTotalVarDict, True, 'mean')
if saveToFile:
    fig2.savefig(folderName + '_weightedMeanStandardDeviation.svg', format='svg')
else:
    plt.show()

"""Plot Median and IQR"""
fig3 = plt.figure(203, figsize=(19.2, 10.8), dpi=100)
plt.title("Median and IQR")
functions.plotOverallPerformance(nIter, simCaseList, rmseDict, totalVarDict, False, 'median')
if saveToFile:
    fig3.savefig(folderName + '_MedianIQR.svg', format='svg')
else:
    plt.show()

"""Plot Mean and standard deviation"""
fig4 = plt.figure(204, figsize=(19.2, 10.8), dpi=100)
plt.title("Mean and Standard Deviation")
functions.plotOverallPerformance(nIter, simCaseList, rmseDict, totalVarDict, False, 'mean')
if saveToFile:
    fig4.savefig(folderName + '_MeanStandardDeviation.svg', format='svg')
else:
    plt.show()

# TODO Enable loading of pickled data instead of simulating

# DONE
# TODO Enable changing of parameters while simulating

# Memory profiling:
# mprof run --include-children python3 simulation.py
# mprof plot --output memory-profile.png
