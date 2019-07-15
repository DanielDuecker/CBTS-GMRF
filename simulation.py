from parameters import par
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy import stats
import functions
import main
import os
import csv
import time
import shutil
import pickle

matplotlib.use('TkAgg')

"""Simulation Options"""
beliefOptions = ['seqBayes']  # 'stkf' 'seqBayes', 'regBayes', 'regBayesTrunc'
controlOptions = ['geist']  #'cbts', 'pi2', 'randomWalk', 'geist'

"""Simulation Options"""
saveToFile = False
nSim = 10
nIter = 500
fieldType = 'random'  # 'peak','sine', 'random' or 'predefined'
temporal = False  # True: time varying field
plot = True
"PI2"
K = [15]
H = [20]
nUpdated = [5]
lambd = [1e-1]
pi2ControlCost = [5]
"CBTS"
branchingFactor = [6]
maxDepth = [3]
kappa = [100]
kappaChildSelection = [1]
UCBRewardFactor = [0.05]
cbtsControlCost = [0.2]
discountFactor = [0.5]

"Initialize lists and dicts"
simCaseList = []
parSettingsList = []
diffMeanDict = {}
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
                                parObject = par(belief,control,fieldType,temporal,plot,nIter,K_i,H_i,nUpdated_i,lambd_i,
                                                pi2ControlCost_i,0,0,0,0,0,0,0)
                                parSettingsList.append(parObject)
                                simCaseList.append(belief + '_' + control + '_' + 'K' + str(K_i).replace('.','p')
                                            + '_' + 'H' + str(H_i).replace('.','p')
                                            + '_' + 'nUpdated' + str(nUpdated_i).replace('.','p')
                                            + '_' + 'lambd' + str(lambd_i).replace('.','p')
                                            + '_' + 'pi2ControlCost' + str(pi2ControlCost_i).replace('.','p'))
        if control == 'cbts':
            for branchingFactor_i in branchingFactor:
                for maxDepth_i in maxDepth:
                    for kappa_i in kappa:
                        for kappaChildSelection_i in kappaChildSelection:
                            for UCBRewardFactor_i in UCBRewardFactor:
                                for cbtsControlCost_i in cbtsControlCost:
                                    for discountFactor_i in discountFactor:
                                        parObject = par(belief, control, fieldType, temporal, plot, nIter, 0, 0,
                                                        0, 0, 0, branchingFactor_i, maxDepth_i, kappa_i,
                                                        kappaChildSelection_i, UCBRewardFactor_i, cbtsControlCost_i,
                                                        discountFactor_i)
                                        parSettingsList.append(parObject)
                                        simCaseList.append(belief + '_' + control
                                            + '_' + 'branchingFactor' + str(branchingFactor_i).replace('.','p')
                                            + '_' + 'maxDepth' + str(maxDepth_i).replace('.','p')
                                            + '_' + 'kappa' + str(kappa_i).replace('.','p')
                                            + '_' + 'kappaChildSelection' + str(kappaChildSelection_i).replace('.','p')
                                            + '_' + 'UCBRewardFactor' + str(UCBRewardFactor_i).replace('.','p')
                                            + '_' + 'cbtsControlCost' + str(cbtsControlCost_i).replace('.','p')
                                            + '_' + 'discountFactor' + str(discountFactor_i).replace('.','p'))
        if control == 'randomWalk':
            parObject = par(belief, control, fieldType, temporal, plot, nIter, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
            parSettingsList.append(parObject)
            simCaseList.append(belief + '_' + control + '_randomWalk')

        if control == 'geist':
            parObject = par(belief, control, fieldType, temporal, plot, nIter, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
            parSettingsList.append(parObject)
            simCaseList.append(belief + '_' + control)


"""Create directory if data should be saved"""
if saveToFile:
    index = 0
    dirpath = os.getcwd()
    folderName = time.strftime("%Y%m%d") + "_" + str(index)
    path = dirpath + "/sim/" + folderName
    while os.path.exists(path):
        index += 1
        folderName = time.strftime("%Y%m%d") + "_" + str(index)
        path = dirpath + "/sim/" + folderName
    try:
        os.mkdir(path)
    except:
        print("Error while creating directory!")
    os.chdir(path)

    # Copy used parameters
    shutil.copyfile(dirpath + '/parameters.py', path + '/' + folderName + '_parameters.txt')

for i in range(len(parSettingsList)):
    par = parSettingsList[i]
    simCase = simCaseList[i]

    """Initialize"""
    x = []
    y = []
    trueField = []
    gmrf = []
    controller = []
    CBTS = []
    timeVec = []
    xHist = []
    yHist = []
    diffMean = []
    totalVar = []

    for i in range(nSim):
        print("Simulation ", i, " of ", nSim, " with ", simCase)
        xR, yR, trueFieldR, gmrfR, controllerR, CBTSR, timeVecR, xHistR, yHistR, diffMeanR, totalVarR = main.main(par)

        x.append(xR)
        y.append(yR)
        trueField.append(trueFieldR)
        gmrf.append(gmrfR)
        controller.append(controllerR)
        CBTS.append(CBTSR)
        timeVec.append(timeVecR)
        xHist.append(xHistR)
        yHist.append(yHistR)
        diffMean.append(diffMeanR)
        totalVar.append(totalVarR)

    diffMeanDict[simCase] = diffMean
    totalVarDict[simCase] = totalVar

    if saveToFile:
        """Save objects"""
        # Save objects
        with open('objs' + '_' + simCase + '.pkl', 'wb') as f:
            pickle.dump(
                [x, y, trueField, gmrf, controller, CBTS, timeVec, xHist, yHist, diffMean, totalVar, par],f)

        # Getting back the objects:
        # with open('objs.pkl','rb') as f:
        #    x, y, trueField, gmrf, controller, CBTS, timeVec, xHist, yHist, diffMean, totalVar,simCase = pickle.load(f)

        # Save data as csv
        with open(folderName + '_' + simCase + '_data.csv', 'w') as dataFile:
            writer = csv.writer(dataFile)
            for i in range(nSim):
                writer.writerow([i])
                writer.writerow(x[i])
                writer.writerow(y[i])
                writer.writerow(timeVec[i])
                writer.writerow(xHist[i])
                writer.writerow(yHist[i])
                writer.writerow(diffMean[i])
                writer.writerow(totalVar[i])
                writer.writerow(gmrf[i].meanCond)
                writer.writerow(gmrf[i].covCond)
                writer.writerow(["-"])
            dataFile.close()

    """Plot fields"""
    fig0 = plt.figure(100, figsize=(19.2, 10.8), dpi=100)
    print("Plotting..")
    for i in range(nSim):
        functions.plotFields(par, fig0, x[i], y[i], trueField[i], gmrf[i], controller[i], CBTS[i],
                             timeVec[i], xHist[i],
                             yHist[i])
        if saveToFile:
            fig0.savefig(folderName + '_' + str(i) + '_' + simCase + '_' + par.fieldType + '.svg',
                         format='svg')
        else:
            plt.show()
        plt.clf()
    plt.close(fig0)

"""Plot Performance"""
fig1 = plt.figure(200,figsize=(19.2,10.8), dpi=100)
x = np.linspace(0,nIter-1,nIter-1)
plt.title('Performance Measurement')
plt.subplot(211)
for sim in simCaseList:
    meanDiff = np.mean(diffMeanDict[sim],axis=0)
    iqrDiff = stats.iqr(diffMeanDict[sim],axis=0)
    plt.plot(x,meanDiff,label = sim)
    plt.plot(x,meanDiff - iqrDiff,'gray')
    plt.plot(x,meanDiff + iqrDiff,'gray')
    plt.fill_between(x,meanDiff - iqrDiff,meanDiff + iqrDiff, cmap='twilight',alpha=0.4)
plt.xlabel('Iteration Index')
plt.ylabel('Difference Between Ground Truth and Belief')
plt.subplot(212)
for sim in simCaseList:
    meanVar = np.mean(totalVarDict[sim],axis=0)
    iqrVar = stats.iqr(totalVarDict[sim],axis=0)
    plt.plot(x,meanVar,label = sim)
    plt.plot(x,meanVar - iqrVar,'gray')
    plt.plot(x,meanVar + iqrVar,'gray')
    plt.fill_between(x,meanVar - iqrVar,meanVar + iqrVar, cmap='twilight',alpha=0.4)
    plt.legend(loc='best')
plt.xlabel('Iteration Index')
plt.ylabel('Total Belief Uncertainty')
if saveToFile:
    fig1.savefig(folderName + '_performance.svg', format='svg')
plt.show()


# TODO Enable loading of pickled data instead of simulating

# DONE
# TODO Enable changing of parameters while simulating

# Memory profiling:
# mprof run --include-children python3 simulation.py
# mprof plot --output memory-profile.png



