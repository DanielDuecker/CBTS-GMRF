# First 2D-GMRF Implementation

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import math
import time
import methods
from parameters import gmrf 

# Parameters
nIter = 100000
oz2 = 0.01              # measurement variance
dX = dY = 0.01          #discretizaton in x and y

# Initialize GMRF
gmrf1=gmrf(0,9,10,0,9,10)

# Set up axes
x = np.arange(gmrf1.xMin,gmrf1.xMax,dX)
y = np.arange(gmrf1.yMin,gmrf1.yMax,dY)
X, Y = np.meshgrid(x,y)

# Ground truth
xGT = np.array([0,2,4,6,9])    # column coordinates
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

##_____GMRF______##
# Precision matrix Q
#Q = np.eye(gmrf1.nP,gmrf1.nP)+np.random.rand(gmrf1.nP,gmrf1.nP)                   #TO DO: Edit Q Matrix
Q = methods.getPrecisionMatrix(gmrf1)

# Plotting initial belief (mean,variance and calculation time)
ax2 = fig.add_subplot(222)
ax2.contourf(gmrf1.x,gmrf1.y,gmrf1.mu.reshape(gmrf1.nX,gmrf1.nY))
plt.xlabel("x in m")
plt.ylabel("y in m")
plt.title("Mean of belief")

ax3 = fig.add_subplot(223)
ax3.contourf(gmrf1.x,gmrf1.y,np.diag(gmrf1.prec).reshape(gmrf1.nX,gmrf1.nY))
plt.xlabel("x in m")
plt.ylabel("y in m")
plt.title("Precision of belief")

timeVec = []
iterVec = []
ax4 = fig.add_subplot(224)
ax4.plot(iterVec,timeVec)
plt.xlabel("Iteration index")
plt.ylabel("calculation time in s")
plt.title("Update calculation time over iteration index")
plt.show()

# Get first measurement:
xMeas = np.random.uniform(gmrf1.xMin,gmrf1.xMax)
yMeas = np.random.uniform(gmrf1.yMin,gmrf1.yMax)
zMeas = np.array([methods.getMeasurement(xMeas,yMeas,f,oz2)])

A = methods.mapConDis(gmrf1,xMeas,yMeas,zMeas[-1])

# Update and plot field belief
for i in range(nIter):
    print("Iteration ",i," of ",nIter,".")
    timeBefore = time.time()

    # Update mean
    temp1 = zMeas - np.dot(A,gmrf1.mu)
    temp2 = np.dot(A.T,temp1)
    gmrf1.mu = gmrf1.mu + 1/oz2*np.dot(np.linalg.inv(gmrf1.prec),temp2)

    # Update precision matrix
    gmrf1.prec = (Q+1/oz2*np.dot(A.T,A))

    # Get next measurement at random position, stack under measurement vector
    xMeas = np.random.uniform(gmrf1.xMin,gmrf1.xMax)
    yMeas = np.random.uniform(gmrf1.yMin,gmrf1.yMax)
    zMeas = np.vstack((zMeas,methods.getMeasurement(xMeas,yMeas,f,oz2)))

    # Map measurement to lattice and stack under A matrix
    A = np.vstack((A,methods.mapConDis(gmrf1,xMeas,yMeas,zMeas[-1])))

    timeAfter = time.time()
    iterVec.append(i)
    timeVec.append(timeAfter-timeBefore)
    print("timeAfter-timeBefore:",timeAfter-timeBefore)

    # Plotting:
    #ax2.contourf(gmrf1.x,gmrf1.y,gmrf1.mu.reshape(gmrf1.nX,gmrf1.nY))
    #ax3.contourf(gmrf1.x,gmrf1.y,np.diag(gmrf1.prec).reshape(gmrf1.nX,gmrf1.nY))
    #ax4.plot(iterVec,timeVec,'black')
    #fig.canvas.draw()
    #fig.canvas.flush_events()

ax2.contourf(gmrf1.x,gmrf1.y,gmrf1.mu.reshape(gmrf1.nX,gmrf1.nY))
ax3.contourf(gmrf1.x,gmrf1.y,np.diag(gmrf1.prec).reshape(gmrf1.nX,gmrf1.nY))
ax4.plot(iterVec,timeVec,'black')
fig.canvas.draw()
fig.canvas.flush_events()

while True:
    wait = True