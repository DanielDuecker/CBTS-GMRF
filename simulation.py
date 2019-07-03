import parameters as par
import numpy as np
import matplotlib.pyplot as plt
import functions
import main
import os
import time
import shutil

saveToFile = True

"""Simulation Parameters"""
belief = 'stkf' # 'seqBayes', 'regBayes'
fieldType = 'predefined'  # 'peak','sine' or 'predefined'
control = 'cbts' # 'pi2'
temporal = False  # True: time varying field

fastCalc = True  # True: Fast Calculation, only one plot in the end; False: Live updating and plotting
truncation = False

nSim = 2
nIter = par.nIter

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
    xR, yR, trueFieldR, gmrfR, controllerR, CBTSR, timeVecR, xHistR, yHistR, diffMeanR, totalVarR = main.main()

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

    # Save data
    f = open(time.strftime("%Y%m%d") + "_" + str(index) + '_data.txt','w+')
    f.close()

    # Plot data
    fig0 = plt.figure(100, figsize=(19.2,10.8), dpi=100)
    for i in range(nSim):
        print("plot fields")
        functions.plotFields(fig0, x[i], y[i], trueField[i], gmrf[i], controller[i], CBTS[i], timeVec[i], xHist[i], yHist[i])
        fig0.savefig(time.strftime("%Y%m%d") + "_" + str(index) + '_' + str(i) + 'fields.svg', format='svg')
        plt.clf()
    plt.close(fig0)

    fig1 = plt.figure(200,figsize=(19.2,10.8), dpi=100)
    plt.title('Performance Measurement')
    ax1 = fig1.add_subplot(211)
    for i in range(nSim):
        ax1.plot(diffMean[i])
    plt.xlabel('Iteration Index')
    plt.ylabel('Difference Between Ground Truth and Belief')
    ax2 = fig1.add_subplot(212)
    for i in range(nSim):
        ax2.plot(totalVar[i])
    plt.xlabel('Iteration Index')
    plt.ylabel('Total Belief Uncertainty')
    plt.show()
    fig1.savefig(time.strftime("%Y%m%d") + "_" + str(index) + '_fig.svg', format='svg')




