import parameters as par
import numpy as np
import matplotlib.pyplot as plt
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
nIter = par.nIter-1

diffMeanMat = np.zeros((nSim,nIter))
totalVarMat = np.zeros((nSim,nIter))

for i in range(nSim):
    print("Simulation ",i," of ",nSim)
    diffMeanMat[i,:],totalVarMat[i,:] = main.main()


fig = plt.figure(0)
plt.title('Performance Measurement')
ax1 = fig.add_subplot(211)
for i in range(nSim):
    ax1.plot(diffMeanMat[i,:])
plt.xlabel('Iteration Index')
plt.ylabel('Difference Between Ground Truth and Belief')
ax2 = fig.add_subplot(212)
for i in range(nSim):
    ax2.plot(totalVarMat[i,:])
plt.xlabel('Iteration Index')
plt.ylabel('Total Belief Uncertainty')

if saveToFile:
    index = 0
    dirpath = os.getcwd()
    path = dirpath + "/sim/" + time.strftime("%Y%m%d") + "_" + str(index)
    while os.path.exists(path):
        index += 1
        path = dirpath + "/sim/" + time.strftime("%Y%m%d") + "_" + str(index)
        print(path)
    try:
        os.mkdir(path)
    except:
        print("Error while creating directory!")

    os.chdir(path)

    # Copy used parameters
    shutil.copyfile(dirpath + '/parameters.py',path + time.strftime("%Y%m%d") + "_" + str(index) + '_parameters.txt')

    # Save data
    f = open(time.strftime("%Y%m%d") + "_" + str(index) + '_data.txt','w+')
    f.close()

    # Save figures
    fig.savefig(time.strftime("%Y%m%d") + "_" + str(index) + '_fig.eps', format='eps')
    fig.savefig(time.strftime("%Y%m%d") + "_" + str(index) + '_fig.jpg', format='jpg')


