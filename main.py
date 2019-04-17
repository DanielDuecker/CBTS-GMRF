# First 2D-GMRF Implementation

import time

import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate

import methods
import parameters as par
from gmrfClass import gmrf

xHist = [par.x0]  # x-state history vector
yHist = [par.y0]  # y-state history vector

# Initialize GMRF   
gmrf1 = gmrf(par.xMin, par.xMax, par.nX, par.yMin, par.yMax, par.nY, par.nBeta)

# Time measurement vectors
timeVec = []
iterVec = []

"""Ground truth"""
# Set up axes
x = np.arange(gmrf1.xMin, gmrf1.xMax, par.dX)
y = np.arange(gmrf1.yMin, gmrf1.yMax, par.dY)
X, Y = np.meshgrid(x, y)

# True field values
xGT = np.array([0, 2, 4, 6, 9])  # column coordinates
yGT = np.array([0, 1, 3, 5, 9])  # row coordinates
zGT = np.array([[1, 2, 2, 1, 1],
                [2, 4, 2, 1, 1],
                [1, 2, 3, 3, 3],
                [1, 3, 2, 1, 4],
                [1, 1, 2, 3, 3]])
f = interpolate.interp2d(xGT, yGT, zGT)

"""GMRF"""
# Initialize Plot
fig = plt.figure()
methods.plotFields(fig, x, y, f, gmrf1, iterVec, timeVec, xHist, yHist)
plt.show()

# Get first measurement:
(xMeas, yMeas) = methods.getNextState(par.x0, par.y0, par.x0, par.y0, par.maxStepsize, gmrf1)
xHist.append(xMeas)
yHist.append(yMeas)
nMeas = 50
zMeas = np.zeros((nMeas,1))
Phi = np.zeros((nMeas,gmrf1.nP+gmrf1.nBeta))
zMeas[0] = methods.getMeasurement(xMeas, yMeas, f, par.ov2)

Phi[0,:] = methods.mapConDis(gmrf1, xMeas, yMeas)

# Update and plot field belief
for i in range(par.nIter):
    print("Iteration ", i, " of ", par.nIter, ".")
    timeBefore = time.time()

    # Bayesian update
    gmrf1.bayesianUpdate(zMeas,Phi)
    #gmrf1.seqBayesianUpdate(zMeas, Phi)

    # Get next measurement according to dynamics, stack under measurement vector
    (xMeas, yMeas) = methods.getNextState(xMeas, yMeas, xHist[-2], yHist[-2], par.maxStepsize, gmrf1)
    xHist.append(xMeas)
    yHist.append(yMeas)
    zMeas[i%nMeas] = methods.getMeasurement(xMeas, yMeas, f, par.ov2)

    # Map measurement to surrounding grid vertices and stack under Phi matrix
    Phi[i%nMeas,:] = methods.mapConDis(gmrf1, xMeas, yMeas)

    if i%nMeas == 0:
        gmrf1.covPrior = gmrf1.covCond
        gmrf1.meanPrior = gmrf1.meanCond

    print(gmrf1.meanCond)

    # Time measurement
    timeAfter = time.time()
    iterVec.append(i)
    timeVec.append(timeAfter - timeBefore)

    # Plotting:
    if not par.fastCalc:
        methods.plotFields(fig, x, y, f, gmrf1, iterVec, timeVec, xHist, yHist)

methods.plotFields(fig, x, y, f, gmrf1, iterVec, timeVec, xHist, yHist)
plt.show(block=True)

print("Last updates needed approx. ", np.mean(timeVec[-100:-1]), " seconds per iteration.")

#TODO augment to time dependent fields
#TODO add outer grid
#TODO boundary conditions
#TODO use of sparse commands
