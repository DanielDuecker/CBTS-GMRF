import parameters as par
import numpy as np
import matplotlib.pyplot as plt
import main

nSim = 10
nIter = par.nIter-1

diffMeanMat = np.zeros((nSim,nIter))
totalVarMat = np.zeros((nSim,nIter))

for i in range(nSim):
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
plt.show(block=True)
