import parameters
import numpy as np
import matplotlib.pyplot as plt
import functions
import main
import os
import csv
import time
import shutil
import pickle
import copy

"""Simulation Parameters"""
saveToFile = True
nSim = 50

"""Get and modify Simulation Parameters"""
par = copy.deepcopy(parameters.par)

par.belief = 'stkf'  # 'stkf' 'seqBayes', 'regBayes', 'regBayesTrunc'
par.control = 'cbts'  #'cbts', 'pi2', 'randomWalk'
par.fieldType = 'predefined'  # 'peak','sine' or 'predefined'
par.temporal = False  # True: time varying field
par.plot = False
parSimList = [par.belief, par.control, par.fieldType, par.temporal]

if par.belief != 'regBayesTrunc':
    par.nMeas = par.nIter

if par.belief == 'stkf':
    par.nBeta = 0

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

for i in range(nSim):
    print("Simulation ",i," of ",nSim)
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

if saveToFile:
    # Create new directory
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
    shutil.copyfile(dirpath + '/parameters.py',path + '/' + folderName + '_parameters.txt')

    """Save objects"""
    # Save objects
    with open('objs.pkl', 'wb') as f:
        pickle.dump([x, y, trueField, gmrf, controller, CBTS, timeVec, xHist, yHist, diffMean, totalVar, parList], f)

    # Getting back the objects:
    #with open('objs.pkl','rb') as f:
    #    x, y, trueField, gmrf, controller, CBTS, timeVec, xHist, yHist, diffMean, totalVar, parList = pickle.load(f)

    # Save data as csv
    with open(folderName + '_data.csv','w') as dataFile:
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

    """Plot data"""
    fig0 = plt.figure(100, figsize=(19.2,10.8), dpi=100)
    print("Plotting..")
    for i in range(nSim):
        functions.plotFields(par,fig0, x[i], y[i], trueField[i], gmrf[i], controller[i], CBTS[i], timeVec[i], xHist[i], yHist[i])
        fig0.savefig(folderName + '_' + str(i) + '_' + par.belief + '_' + par.control + '_' + par.fieldType +'.svg', format='svg')
        plt.clf()
    plt.close(fig0)

    fig1 = plt.figure(200,figsize=(19.2,10.8), dpi=100)
    plt.title('Performance Measurement')
    plt.subplot(211)
    for i in range(nSim):
        plt.plot(diffMean[i])
    plt.xlabel('Iteration Index')
    plt.ylabel('Difference Between Ground Truth and Belief')
    plt.subplot(212)
    for i in range(nSim):
        plt.plot(totalVar[i])
    plt.xlabel('Iteration Index')
    plt.ylabel('Total Belief Uncertainty')
    fig1.savefig(folderName + '_performance.svg', format='svg')

# TODO Enable loading of pickled data instead of simulating
# TODO Enable changing of parameters while simulating



