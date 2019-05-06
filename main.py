# First 2D-GMRF Implementation

import time

import matplotlib.pyplot as plt
import numpy as np

import methods
import parameters as par
from classes import gmrf
from classes import stkf
from classes import trueField

np.set_printoptions(threshold=np.inf)

xHist = [par.x0]  # x-state history vector
yHist = [par.y0]  # y-state history vector

# Time measurement vectors
timeVec = []
iterVec = []

# Initialize GMRF
gmrf1 = gmrf(par.xMin, par.xMax, par.nX, par.yMin, par.yMax, par.nY, par.nBeta)

# Plotting grid
x = np.arange(gmrf1.xMin, gmrf1.xMax, par.dX)
y = np.arange(gmrf1.yMin, gmrf1.yMax, par.dY)
X, Y = np.meshgrid(x, y)

"""Ground Truth"""
trueField = trueField(x[-1], y[-1], par.sinusoidal, par.temporal)

"""STKF"""
stkf1 = stkf(par.xMin, par.xMax, par.nX, par.yMin, par.yMax, par.nY, par.nBeta, trueField, par.dt, par.sigmaT,
             par.lambd, par.sigma2)

"""GMRF"""
# Initialize Plot
fig = plt.figure()
methods.plotFields(fig, x, y, trueField, gmrf1, iterVec, timeVec, xHist, yHist)
plt.show()

# Get first measurement:
(xMeas, yMeas) = methods.getNextState(par.x0, par.y0, par.x0, par.y0, par.maxStepsize, gmrf1)
xHist.append(xMeas)
yHist.append(yMeas)

# Initialize measurement vector and mapping matrix
zMeas = np.zeros((par.nMeas, 1))
Phi = np.zeros((par.nMeas, gmrf1.nP + gmrf1.nBeta))
zMeas[0] = methods.getMeasurement(xMeas, yMeas, trueField, par.ov2)
Phi[0, :] = methods.mapConDis(gmrf1, xMeas, yMeas)

# Update and plot field belief
for i in range(par.nIter - 1):
    print("Iteration ", i, " of ", par.nIter, ".")
    t = i * par.dt

    timeBefore = time.time()

    # Bayesian update
    if par.stkf:
        stkf1.kalmanFilter(t, xMeas, yMeas, zMeas[i])
        gmrf1.meanCond = stkf1.gmrf.meanCond
        gmrf1.covCond = stkf1.gmrf.covCond
        gmrf1.diagCovCond = stkf1.gmrf.diagCovCond
    else:
        if par.sequentialUpdate:
            gmrf1.seqBayesianUpdate(zMeas[i], Phi[i, :])
        else:
            gmrf1.bayesianUpdate(zMeas[0:i], Phi[0:i, :])

    # Get next measurement according to dynamics, stack under measurement vector
    (xMeas, yMeas) = methods.getNextState(xMeas, yMeas, xHist[-2], yHist[-2], par.maxStepsize, gmrf1)
    xHist.append(xMeas)
    yHist.append(yMeas)
    zMeas[(i + 1) % par.nMeas] = methods.getMeasurement(xMeas, yMeas, trueField, par.ov2)

    # Map measurement to surrounding grid vertices and stack under Phi matrix
    Phi[(i + 1) % par.nMeas, :] = methods.mapConDis(gmrf1, xMeas, yMeas)

    # If truncated measurements are used, set conditioned mean and covariance as prior
    if par.truncation:
        if (i + 1) % par.nMeas == 0:
            gmrf1.covPrior = gmrf1.covCond
            gmrf1.meanPrior = gmrf1.meanCond

    # Time measurement
    timeAfter = time.time()
    iterVec.append(i)
    timeVec.append(timeAfter - timeBefore)

    # Plotting:
    if not par.fastCalc:
        #if i % (par.nIter/100) == 0:
        methods.plotFields(fig, x, y, trueField, gmrf1, iterVec, timeVec, xHist, yHist)

    # Update ground truth:
    if par.temporal:
        trueField.updateField(i)

methods.plotFields(fig, x, y, trueField, gmrf1, iterVec, timeVec, xHist, yHist)
plt.show(block=True)

print("Last updates needed approx. ", np.mean(timeVec[-100:-1]), " seconds per iteration.")

# TODO add outer grid
# TODO boundary conditions
# TODO use of sparse commands
