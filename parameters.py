import numpy as np
import math

# Parameters for all files
# main.py
stkf = True
sequentialUpdate = True  # Does not work with truncation!
fastCalc = False  # True: Fast Calculation, only one plot in the end; False: Live updating and plotting
truncation = False
sinusoidal = False  # True: Use sinusoidal ground truth
temporal = False  # True: time varying field
PIControl = False

if not PIControl:
    CBTS = True
else:
    CBTS = False

exploitingRate = 0

nIter = 1000  # number of iterations
dt = 1  # timesteps per iteration
nMeas = 100  # number of measurements for bayesian inference (nMeas = nIter for inference without truncation)
ov2 = 0.04  # measurement variance
dX = dY = 0.01  # discretizaton in x and y for Plotting

(x0, y0, alpha0) = (0, 0, math.pi/4)  # initial state
maxStepsize = 0.1  # maximum change in every state per iteration
xVel = maxStepsize
yVel = maxStepsize

dxdt = 0.001  # Shift of true field in x direction
dydt = 0.001  # Shift of true field in y direction
pulseTime = nIter / 2  # Duration of sinusodial pulsation

xMin = 0  # GMRF dimensions
xMax = 10
nX = 10
yMin = 0
yMax = 10
nY = 10
nBeta = 1  # regression coefficients

# gmrf class
valueT = 1e-3  # Prior precision value for regression vector bet

# stkf class
sigmaT = 1e3 #1e2    # determines exponential decay of time kernel
lambdSTKF = 1  # influences time kernel value
sigma2 = 0.01

# PI2 controller
H = 30  # control horizon steps
controlCost = 5# 5e-1   # affects noise of path roll-outs (negatively)
R = controlCost*np.eye(H)   # input cost matrix
g = np.ones((H,1))
lambd = 1e-1 # 1e-2 #
# rescales state costs, affects noise of path roll-outs (positively)
K = 15  # number of path roll outs
ctrSamplingTime = 0.01  # time discretization
nUpdated = 10   # number of iterations

# CBTS controller
trajStepSize = 1
CBTSIterations = 30
nTrajPoints = int(trajStepSize/maxStepsize)
maxParamExploration = 3
trajOrder = 3
maxDepth = 2
branchingFactor = 10
kappa = 1

# action reward map
ovMap2 = 0.01

if not truncation:
    nMeas = nIter

if stkf:
    nBeta = 0
