# First 2D-GMRF Implementation

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import math
import time
import methods
from parameters import gmrf 

# Parameters
nIter = 10000
oz2 = 0.01              # measurement variance
dX = dY = 0.01          #discretizaton in x and y

# Initialize GMRF
gmrf1=gmrf(0,10,20,0,10,20)

# Set up axes
x = np.arange(gmrf1.xMin,gmrf1.xMax,dX)
y = np.arange(gmrf1.yMin,gmrf1.yMax,dY)
X, Y = np.meshgrid(x,y)

# Ground truth
xGT = np.array([-4,-2,0,2,4])    # column coordinates
yGT =  np.array([-4,-2,0,2,4])    # row coordinates
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
Q = np.eye(gmrf1.nP,gmrf1.nP)+np.random.rand(gmrf1.nP,gmrf1.nP)                   #TO DO: Edit Q Matrix
#Q = np.eye(nY*nX)

# Plotting initial belief (mean and variance)
ax2 = fig.add_subplot(222)
ax2.contourf(gmrf1.x,gmrf1.y,gmrf1.mu.reshape(gmrf1.nX,gmrf1.nY))
plt.title("Mean of belief")

ax3 = fig.add_subplot(223)
ax3.contourf(gmrf1.x,gmrf1.y,np.diag(gmrf1.prec).reshape(gmrf1.nX,gmrf1.nY))
plt.title("Variance of belief")

plt.show()

# Get first measurement:
xMeas = np.random.uniform(gmrf1.xMin,gmrf1.xMax)
yMeas = np.random.uniform(gmrf1.yMin,gmrf1.yMax)
zMeas = np.array([methods.getMeasurement(xMeas,yMeas,f,oz2)])

A = methods.mapConDis(gmrf1,xMeas,yMeas,zMeas[-1])

#Update and plot field belief
for i in range(nIter):
    print("Iteration ",i," of ",nIter,".")

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

    # Plotting:
    ax2.contourf(gmrf1.x,gmrf1.y,gmrf1.mu.reshape(gmrf1.nX,gmrf1.nY))
    ax3.contourf(gmrf1.x,gmrf1.y,np.diag(gmrf1.prec).reshape(gmrf1.nX,gmrf1.nY))
    fig.canvas.draw()
    fig.canvas.flush_events()
    print("New mean: \n",gmrf1.mu)