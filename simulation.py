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

"""Simulation Parameters"""
saveToFile = True
nSim = 2


"""Get and modify Simulation Parameters"""
par = parameters.par

par.belief = 'stkf'  # 'seqBayes', 'regBayes', 'regBayesTrunc'
par.control = 'cbts'  # 'pi2', 'randomWalk'
fieldType = 'predefined'  # 'peak','sine' or 'predefined'
temporal = False  # True: time varying field
plot = False

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

if saveToFile:
    # Create new directory
    index = 0
    dirpath = os.getcwd()
    path = dirpath + "/sim/" + time.strftime("%Y%m%d") + "_" + str(index)
    while os.path.exists(path):
        index += 1
        path = dirpath + "/sim/" + time.strftime("%Y%m%d") + "_" + str(index)
    try:
        os.mkdir(path)
    except:
        print("Error while creating directory!")
    os.chdir(path)

    # Copy used parameters
    shutil.copyfile(dirpath + '/parameters.py',path + time.strftime("/%Y%m%d") + "_" + str(index) + '_parameters.txt')

    """Save objects"""
    # Save objects
    with open('objs.pkl', 'wb') as f:
        pickle.dump([x, y, trueField, gmrf, controller, CBTS, timeVec, xHist, yHist, diffMean, totalVar], f)

    # Getting back the objects:
    #with open('objs.pkl','rb') as f:
    #    x, y, trueField, gmrf, controller, CBTS, timeVec, xHist, yHist, diffMean, totalVar = pickle.load(f)

    # Save data as csv
    with open(time.strftime("%Y%m%d") + "_" + str(index) + '_data.csv','w') as dataFile:
        writer = csv.writer(dataFile)
        for i in range(nSim):
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

    """Plot data"""
    fig0 = plt.figure(100, figsize=(19.2,10.8), dpi=100)
    print("Plotting..")
    for i in range(nSim):
        functions.plotFields(par,fig0, x[i], y[i], trueField[i], gmrf[i], controller[i], CBTS[i], timeVec[i], xHist[i], yHist[i])
        fig0.savefig(time.strftime("%Y%m%d") + "_" + str(index) + '_' + str(i) + 'fields.svg', format='svg')
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
    fig1.savefig(time.strftime("%Y%m%d") + "_" + str(index) + '_fig.svg', format='svg')




