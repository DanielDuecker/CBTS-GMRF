import parameters
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
import copy

matplotlib.use('TkAgg')

"""Load Parameters"""
par = copy.deepcopy(parameters.par)
"""Simulation Options"""
beliefOptions = ['stkf']  # 'stkf' 'seqBayes', 'regBayes', 'regBayesTrunc'
controlOptions = ['cbts']  #'cbts', 'pi2', 'randomWalk'

saveToFile = True
nSim = 1
par.nIter = 1000
par.fieldType = 'sine'  # 'peak','sine' or 'predefined'
par.temporal = False  # True: time varying field
par.plot = True

parSimList = [nSim, par.nIter, par.belief, par.control, par.fieldType, par.temporal]
simList = []
diffMeanDict = {}
totalVarDict = {}

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

"""Iterate Through Simulation Options"""
for belief in beliefOptions:
    par.belief = belief
    for control in controlOptions:
        par.control = control
        simCase = belief + '_' + control
        simList.append(simCase)

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
        parList = []

        if par.belief != 'regBayesTrunc':
            par.nMeas = par.nIter
        if par.belief == 'stkf':
            par.nBeta = 0

        for i in range(nSim):
            print("Simulation ",i," of ",nSim," with ",par.belief, " belief and controller ", par.control)
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
            parList.append(par)

        diffMeanDict[simCase] = diffMean
        totalVarDict[simCase] = totalVar

        if saveToFile:
            # Copy used parameters
            shutil.copyfile(dirpath + '/parameters.py',path + '/' + folderName + '_' + simCase + '_parameters.txt')

            """Save objects"""
            # Save objects
            with open('objs' + '_' + simCase + '.pkl', 'wb') as f:
                pickle.dump([x, y, trueField, gmrf, controller, CBTS, timeVec, xHist, yHist, diffMean, totalVar, parList], f)

            # Getting back the objects:
            # with open('objs.pkl','rb') as f:
            #    x, y, trueField, gmrf, controller, CBTS, timeVec, xHist, yHist, diffMean, totalVar, parList = pickle.load(f)

            # Save data as csv
            with open(folderName + '_' + simCase + '_data.csv','w') as dataFile:
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
                    writer.writerow(parSimList)
                    writer.writerow(["-"])
                dataFile.close()

        """Plot fields"""
        fig0 = plt.figure(100, figsize=(19.2,10.8), dpi=100)
        print("Plotting..")
        for i in range(nSim):
            functions.plotFields(par,fig0, x[i], y[i], trueField[i], gmrf[i], controller[i], CBTS[i], timeVec[i], xHist[i], yHist[i])
            if saveToFile:
                fig0.savefig(folderName + '_' + str(i) + '_' + simCase + '_' + par.fieldType + '.svg', format='svg')
            else:
                plt.show()
            plt.clf()
        plt.close(fig0)

fig1 = plt.figure(200,figsize=(19.2,10.8), dpi=100)
x = np.linspace(0,par.nIter-1,par.nIter-1)
plt.title('Performance Measurement')
plt.subplot(211)
for sim in simList:
    meanDiff = np.mean(diffMeanDict[sim],axis=0)
    iqrDiff = stats.iqr(diffMeanDict[sim],axis=0)
    plt.plot(x,meanDiff,label = sim)
    plt.plot(x,meanDiff - iqrDiff,'gray')
    plt.plot(x,meanDiff + iqrDiff,'gray')
    plt.fill_between(x,meanDiff - iqrDiff,meanDiff + iqrDiff, cmap='twilight',alpha=0.4)
    plt.legend()
plt.xlabel('Iteration Index')
plt.ylabel('Difference Between Ground Truth and Belief')
plt.subplot(212)
for sim in simList:
    meanVar = np.mean(totalVarDict[sim],axis=0)
    iqrVar = stats.iqr(totalVarDict[sim],axis=0)
    plt.plot(x,meanVar,label = sim)
    plt.plot(x,meanVar - iqrVar,'gray')
    plt.plot(x,meanVar + iqrVar,'gray')
    plt.fill_between(x,meanVar - iqrVar,meanVar + iqrVar, cmap='twilight',alpha=0.4)
    plt.legend()
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



