# First 2D-GMRF Implementation

import time

import matplotlib.pyplot as plt
import numpy as np

import methods
import parameters as par
from classes import gmrf
from classes import stkf
from classes import trueField
import control
from CBTS import CBTS

#np.set_printoptions(threshold=np.inf)

"""Agent"""
auv = control.agent(par.x0, par.y0, par.alpha0)
xHist = [par.x0]  # x-state history vector
yHist = [par.y0]  # y-state history vector

"""Plotting grid"""
x = np.arange(par.xMin, par.xMax, par.dX)
y = np.arange(par.yMin, par.yMax, par.dY)
X, Y = np.meshgrid(x, y)

"""Time"""
timeVec = []
iterVec = []

"""GMRF representation"""
gmrf1 = gmrf(par.xMin, par.xMax, par.nX, par.yMin, par.yMax, par.nY, par.nBeta)

"""PI2 Controller"""
controller = control.piControl(par.R, par.g, par.lambd, par.H, par.K, par.ctrSamplingTime, par.nUpdated)

"""Ground Truth"""
trueField = trueField(x[-1], y[-1], par.sinusoidal, par.temporal)

"""STKF extension of gmrf"""
stkf1 = stkf(gmrf1, trueField, par.dt, par.sigmaT, par.lambdSTKF, par.sigma2)

""""Continuous Belief Tree Search"""
CBTS1 = CBTS(par.CBTSIterations, par.nTrajPoints, par.trajOrder, par.maxDepth, par.branchingFactor, par.kappa)
bestTraj = np.zeros((2,1))

"""Initialize plot"""
fig = plt.figure(0)
methods.plotFields(fig, x, y, trueField, gmrf1, controller,CBTS1, iterVec, timeVec, xHist, yHist)
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

    """Update belief"""
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

    """Controller"""
    if par.PIControl:
        # Get next state according to PI Controller
        xMeas, yMeas = controller.getNewState(auv, gmrf1)
    elif par.CBTS:
        if i%par.nTrajPoints == 0:
            test=1
            bestTraj,auv.derivX, auv.derivY = CBTS1.getNewTraj(auv,gmrf1)
            #print("New trajectory generated:", bestTraj)
        auv.x = bestTraj[0,i%par.nTrajPoints]
        auv.y = bestTraj[1,i%par.nTrajPoints]
        xMeas = auv.x
        yMeas = auv.y
    else:
        # Get next measurement according to dynamics, stack under measurement vector
        xMeas, yMeas = methods.getNextState(xMeas, yMeas, xHist[-2], yHist[-2], par.maxStepsize, gmrf1)

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
        methods.plotFields(fig, x, y, trueField, gmrf1, controller,CBTS1, iterVec, timeVec, xHist, yHist)

    # Update ground truth:
    if par.temporal:
        trueField.updateField(i)

methods.plotFields(fig, x, y, trueField, gmrf1, controller,CBTS1, iterVec, timeVec, xHist, yHist)
plt.show(block=True)

print("Last updates needed approx. ", np.mean(timeVec[-100:-1]), " seconds per iteration.")

# TODO Check visit counter
# TODO maybe discount future rewards for exploration
# TODO Check computation time
# TODO -> Change implementation of belief update at each node
# TODO tidy up code, consistent classes and paramater policy
# TODO Try small thetas for less branching and deeper trees
# TODO Use generic gmrf implementation (also for action reward mapping)
# TODO Use current belief mean in reward function -> more exploitation
# TODO maybe use cubic splines or kernel trajs
# TODO add outer grid
# TODO boundary conditions
# TODO use of sparse commands
