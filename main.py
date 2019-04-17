# First 2D-GMRF Implementation

import time

import matplotlib.pyplot as plt
import numpy as np

import methods
import parameters as par
from classes import gmrf
from classes import trueField

xHist = [par.x0]  # x-state history vector
yHist = [par.y0]  # y-state history vector

# Initialize GMRF   
gmrf1 = gmrf(par.xMin, par.xMax, par.nX, par.yMin, par.yMax, par.nY, par.nBeta)

# Time measurement vectors
timeVec = []
iterVec = []

x = np.arange(gmrf1.xMin, gmrf1.xMax, par.dX)
y = np.arange(gmrf1.yMin, gmrf1.yMax, par.dY)
X, Y = np.meshgrid(x, y)

"""Ground Truth"""
groundTruth = trueField()

"""GMRF"""
# Initialize Plot
fig = plt.figure()
methods.plotFields(fig, x, y, groundTruth, 0, gmrf1, iterVec, timeVec, xHist, yHist)
plt.show()

# Get first measurement:
(xMeas, yMeas) = methods.getNextState(par.x0, par.y0, par.x0, par.y0, par.maxStepsize, gmrf1)
xHist.append(xMeas)
yHist.append(yMeas)
zMeas = np.array([methods.getMeasurement(xMeas, yMeas, groundTruth.f, par.ov2)])

Phi = methods.mapConDis(gmrf1, xMeas, yMeas)

# Update and plot field belief
for i in range(par.nIter):
    print("Iteration ", i, " of ", par.nIter, ".")
    timeBefore = time.time()

    # Bayesian update
    # gmrf1.bayesianUpdate(zMeas,Phi)
    gmrf1.seqBayesianUpdate(zMeas, Phi)

    # Get next measurement according to dynamics, stack under measurement vector
    (xMeas, yMeas) = methods.getNextState(xMeas, yMeas, xHist[-2], yHist[-2], par.maxStepsize, gmrf1)
    xHist.append(xMeas)
    yHist.append(yMeas)
    zMeas = np.vstack((zMeas, methods.getMeasurement(xMeas, yMeas, groundTruth.f, par.ov2)))

    # Map measurement to surrounding grid vertices and stack under Phi matrix
    Phi = np.vstack((Phi, methods.mapConDis(gmrf1, xMeas, yMeas)))

    # Time measurement
    timeAfter = time.time()
    iterVec.append(i)
    timeVec.append(timeAfter - timeBefore)

    # Plotting:
    if not par.fastCalc:
        methods.plotFields(fig, x, y, groundTruth, i, gmrf1, iterVec, timeVec, xHist, yHist)

methods.plotFields(fig, x, y, groundTruth, i, gmrf1, iterVec, timeVec, xHist, yHist)
plt.show(block=True)

print("Last updates needed approx. ", np.mean(timeVec[-100:-1]), " seconds per iteration.")

#TODO augment to time dependent fields
#TODO add outer grid
#TODO boundary conditions
#TODO use of sparse commands
