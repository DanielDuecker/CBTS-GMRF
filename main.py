# First 2D-GMRF Implementation

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import math
import time
import methods
from parameters import gmrf 

## Configuration ##
# Parameters
nIter = 10000                    # number of iterations
oz2 = 0.01                      # measurement variance
dX = dY = 0.01                  # discretizaton in x and y

# State dynamics
(x0,y0) = (0,0)                 # initial state
xHist= [x0]                     # x-state history vector
yHist = [y0]                    # y-state history vector
stepsize = 0.5                  # change in every state per iteration

# Initialize GMRF   
gmrf1=gmrf(0,10,10,0,10,10)       # gmrf1=gmrf(xMin,xMax,nX,yMin,yMax,nY), xMin and xMax need to be positive!

# Time measurement vectors
timeVec = []
iterVec = []


## Ground truth ##
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

# Plotting ground truth
plt.ion()
fig = plt.figure()
ax1 = fig.add_subplot(221)
ax1.contourf(x,y,f(x,y))
plt.title("True field")


## GMRF ##
# Precision matrix Q
Q = methods.getPrecisionMatrix(gmrf1)

# Plotting initial belief (mean,variance and calculation time)
ax2 = fig.add_subplot(222)
ax2.contourf(gmrf1.x,gmrf1.y,gmrf1.muCond.reshape(gmrf1.nX,gmrf1.nY))
plt.xlabel("x in m")
plt.ylabel("y in m")
plt.title("Mean of belief")

ax3 = fig.add_subplot(223)
ax3.contourf(gmrf1.x,gmrf1.y,np.diag(gmrf1.precCond).reshape(gmrf1.nX,gmrf1.nY))
plt.xlabel("x in m")
plt.ylabel("y in m")
plt.title("Precision of belief")

ax4 = fig.add_subplot(224)
ax4.plot(iterVec,timeVec)
plt.xlabel("Iteration index")
plt.ylabel("calculation time in s")
plt.title("Update calculation time over iteration index")
plt.show()

# Get first measurement:
(xMeas,yMeas) = methods.getNextState(x0,y0,stepsize,gmrf1)
zMeas = np.array([methods.getMeasurement(xMeas,yMeas,f,oz2)])

A = methods.mapConDis(gmrf1,xMeas,yMeas,zMeas[-1])

# Update and plot field belief
for i in range(nIter):
    print("Iteration ",i," of ",nIter,".")
    timeBefore = time.time()

    # Update mean           # TO DO: replace muCond with mean of measurements

    # Constant mu
    mu = np.array([np.ones((gmrf1.nY,gmrf1.nX)).flatten()]).T

    # Use conditional mean from last iteration
    #mu = gmrf1.muCond

    temp1 = zMeas - np.dot(A,mu)
    temp2 = np.dot(A.T,temp1)
    gmrf1.muCond = mu + 1/oz2*np.dot(np.linalg.inv(gmrf1.precCond),temp2)

    # Update precision matrix
    gmrf1.precCond = (Q+1/oz2*np.dot(A.T,A))

    # Get next measurement according to dynamics, stack under measurement vector
    (xMeas,yMeas) = methods.getNextState(xMeas,yMeas,stepsize,gmrf1)
    xHist.append(xMeas)
    yHist.append(yMeas)
    zMeas = np.vstack((zMeas,methods.getMeasurement(xMeas,yMeas,f,oz2)))

    # Map measurement to surrounding grid vertices and stack under A matrix
    A = np.vstack((A,methods.mapConDis(gmrf1,xMeas,yMeas,zMeas[-1])))

    # Time measurement
    timeAfter = time.time()
    iterVec.append(i)
    timeVec.append(timeAfter-timeBefore)

    # Update plot:
    #ax2.contourf(gmrf1.x,gmrf1.y,gmrf1.muCond.reshape(gmrf1.nX,gmrf1.nY))
    #ax3.contourf(gmrf1.x,gmrf1.y,np.diag(gmrf1.precCond).reshape(gmrf1.nX,gmrf1.nY))
    #ax4.plot(iterVec,timeVec,'black')
    #fig.canvas.draw()
    #fig.canvas.flush_events()

ax2.contourf(gmrf1.x,gmrf1.y,gmrf1.muCond.reshape(gmrf1.nX,gmrf1.nY))
ax3.contourf(gmrf1.x,gmrf1.y,np.diag(gmrf1.precCond).reshape(gmrf1.nX,gmrf1.nY))
ax3.plot(xHist,yHist,'black')
ax4.plot(iterVec,timeVec,'black')
fig.canvas.draw()
fig.canvas.flush_events()

print("Last updates needed approx. ",np.mean(timeVec[-100:-1])," seconds per iteration.")

plt.show(block=True)
