# First 2D-GMRF Implementation

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import math
import time
import methods
from gmrfClass import gmrf 

"Configuration"
# Parameters
nIter = 1000                    # number of iterations
ov2 = 0.01                      # measurement variance
dX = dY = 0.01                  # discretizaton in x and y
fastCalc = True                 # True: Fast Calculation, only one plot in the end; False: Live updating and plotting

# State dynamics
(x0,y0) = (0,0)                 # initial state
xHist= [x0]                     # x-state history vector
yHist = [y0]                    # y-state history vector
maxStepsize = 0.5               # maximum change in every state per iteration

# Initialize GMRF   
gmrf1=gmrf(0,10,10,0,10,10,1)     # gmrf1=gmrf(self,xMin,xMax,nX,yMin,yMax,nY,nBeta), xMin and xMax need to be positive!

# Time measurement vectors
timeVec = []
iterVec = []

"Ground truth"
# Set up axes
x = np.arange(gmrf1.xMin,gmrf1.xMax,dX)
y = np.arange(gmrf1.yMin,gmrf1.yMax,dY)
X, Y = np.meshgrid(x,y)

# True field values
xGT = np.array([0,2,4,6,9])     # column coordinates
yGT =  np.array([0,1,3,5,9])    # row coordinates
zGT = np.array([[1,2,2,1,1],
                [2,4,2,1,1],
                [1,2,3,3,2],
                [1,1,2,3,3],
                [1,1,2,3,3]])
f = interpolate.interp2d(xGT,yGT,zGT)

"GMRF"
# Initialize Plot
fig = plt.figure()
methods.plotFields(fig,x,y,f,gmrf1,iterVec,timeVec,xHist,yHist)
plt.show()

# Get first measurement:
(xMeas,yMeas) = methods.getNextState(x0,y0,x0,y0,maxStepsize,gmrf1)
xHist.append(xMeas)
yHist.append(yMeas)
zMeas = np.array([methods.getMeasurement(xMeas,yMeas,f,ov2)])

Phi = methods.mapConDis(gmrf1,xMeas,yMeas,zMeas[-1])

# Update and plot field belief
for i in range(nIter):
    print("Iteration ",i," of ",nIter,".")
    timeBefore = time.time()

    # Bayesian update
    gmrf1.bayesianUpdate(zMeas,ov2,Phi)

    # Get next measurement according to dynamics, stack under measurement vector
    (xMeas,yMeas) = methods.getNextState(xMeas,yMeas,xHist[-2],yHist[-2],maxStepsize,gmrf1)
    xHist.append(xMeas)
    yHist.append(yMeas)
    zMeas = np.vstack((zMeas,methods.getMeasurement(xMeas,yMeas,f,ov2)))

    # Map measurement to surrounding grid vertices and stack under Phi matrix
    Phi = np.vstack((Phi,methods.mapConDis(gmrf1,xMeas,yMeas,zMeas[-1])))

    # Time measurement
    timeAfter = time.time()
    iterVec.append(i)
    timeVec.append(timeAfter-timeBefore)

    # Plotting:
    if fastCalc == False:
        methods.plotFields(fig,x,y,f,gmrf1,iterVec,timeVec,xHist,yHist)

methods.plotFields(fig,x,y,f,gmrf1,iterVec,timeVec,xHist,yHist)
plt.show(block=True)

print("Last updates needed approx. ",np.mean(timeVec[-100:-1])," seconds per iteration.")
