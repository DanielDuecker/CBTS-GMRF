import numpy as np
import math

"""Parameters for all files"""

"""main.py"""
stkf = True
sequentialUpdate = True  # Does not work with truncation!
fastCalc = False  # True: Fast Calculation, only one plot in the end; False: Live updating and plotting
truncation = False
sinusoidal = False  # True: Use sinusoidal ground truth
temporal = False # True: time varying field
PIControl = False

if not PIControl:
    CBTS = True
else:
    CBTS = False

exploitingRate = 0

nIter = 1000  # number of iterations
dt = 1  # timesteps per iteration
nMeas = 100  # number of measurements for bayesian inference (nMeas = nIter for inference without truncation)
ov2 = 0.01  # measurement variance
dX = dY = 0.01  # discretizaton in x and y for Plotting

(x0, y0, alpha0) = (0, 0, math.pi/4)  # initial state
maxStepsize = 0.1  # maximum change in every state per iteration
xVel = maxStepsize
yVel = maxStepsize

dxdt = 0.001  # Shift of true field in x direction
dydt = 0.001  # Shift of true field in y direction
pulseTime = nIter / 2  # Duration of sinusodial pulsation

"""GMRF class"""
xMin = 0  # GMRF dimensions
xMax = 10
nX = 20
yMin = 0
yMax = 10
nY = 20
nBeta = 1  # regression coefficients
valueT = 1e-3  # Prior precision value for regression vector bet

"""STKF class"""
sigmaT = 1e3 #1e2    # determines exponential decay of time kernel
lambdSTKF = 1  # influences time kernel value
sigma2 = 0.01

"""PI2 controller"""
H = 10  # control horizon steps
controlCost = 5# 5e-1   # affects noise of path roll-outs (negatively)
R = controlCost*np.eye(H)   # input cost matrix
g = np.ones((H,1))
lambd = 1e-1 # 1e-2 # rescales state costs, affects noise of path roll-outs (positively)
K = 15  # number of path roll outs
ctrSamplingTime = 0.01  # time discretization
nUpdated = 10   # number of iterations
outOfGridPenalty = 10

"""CBTS controller"""
trajStepSize = 0.4
CBTSIterations = 4
branchingFactor = 5 # number of actions that should be evaluated for each path segment
kappa = 100  # large: evaluate more untried actions; small: concentrate on actions which already lead to high rewards
nTrajPoints = int(trajStepSize/maxStepsize)
kappaChildSelection = 0.1

thetaMin = -1
thetaMax = 1
thetaExpMin = -0.2
thetaExpMax = 0.2
trajOrder = 1
maxDepth = 3
initialTheta = np.zeros(trajOrder)
discountFactor = 0.5

# action reward map
ovMap2 = 0.01
kernelPar = 10
nThetaSamples = 100

if not truncation:
    nMeas = nIter

if stkf:
    nBeta = 0

class plotOptions:
    showExploredPaths = True
    showActionRewardMapping = True
