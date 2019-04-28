# Parameter

# main.py
stkf = True
sequentialUpdate = False # Does not work with truncation!
fastCalc = True # True: Fast Calculation, only one plot in the end; False: Live updating and plotting
truncation = False
sinusoidal = True # True: Use sinusoidal ground truth
temporal = False # True: time varying field

nIter = 10000 # number of iterations
dt = 1 # timestep per iteration
nMeas = 100 # number of measurements for bayesian inference (nMeas = nIter for inference without truncation)
ov2 = 0.01 # measurement variance
dX = dY = 0.01  # discretizaton in x and y for Plotting

(x0, y0) = (0, 0)  # initial state
maxStepsize = 1  # maximum change in every state per iteration

dxdt = 0.0001  # Shift of true field in x direction
dydt = 0.0001  # Shift of true field in y direction
pulseTime = nIter/2 # Duration of sinusodial pulsation

xMin = 0  # GMRF dimensions
xMax = 10
nX = 10
yMin = 0
yMax = 10
nY = 10
nBeta = 0  # regression coefficients

# gmrf class
valueT = 1e-3  # Prior precision value for regression vector bet

# stkf class
sigmaT = 1e1
lambd = 1
sigma2 = 0.01

if not truncation:
    nMeas = nIter

if stkf:
    nBeta = 0
