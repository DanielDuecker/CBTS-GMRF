# First 2D-GMRF Implementation

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import math
import time
import methods

# Parameters
nIter = 10000
oz2 = 0.01           # measurement variance

# Set up axes
x = y = np.arange(-4,4,0.01)
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
ax1 = fig.add_subplot(121)
ax1.contourf(x,y,f(x,y))

# Initial GMRF
nY,nX = zGT.shape                   #CAN BE CHANGED
mu = np.array([np.zeros((nY,nX)).flatten()]).T
Q = 2*np.random.rand(nY*nX,nY*nX)                   #TO DO: Edit Q Matrix

# Plotting initial belief
ax2 = fig.add_subplot(122)
ax2.contourf(xGT,yGT,mu.reshape(nY,nX))

plt.show()

# Get first measurement:
xMeas = np.random.choice(xGT)
yMeas = np.random.choice(yGT)
zMeas = np.array([methods.getMeasurement(xMeas,yMeas,f,oz2)])
iPos = int((yMeas/2+2)*5+xMeas/2+3)
A = np.array([np.eye(nY*nX)[iPos,:]])

#Update and plot field belief
for i in range(nIter):

    # Update on sigma
    QCond = (Q+1/oz2*np.dot(A.T,A))

    # Update on mu
    temp1 = zMeas - np.dot(A,mu)
    temp2 = np.dot(A.T,temp1)
    test = mu + 1/oz2*np.dot(np.linalg.inv(QCond),temp2)

    # Get next measurement at random position, stack under measurement vector
    xMeas = np.random.choice(xGT)
    yMeas = np.random.choice(yGT)
    zMeas = np.vstack((zMeas,methods.getMeasurement(xMeas,yMeas,f,oz2)))
    
    # Map measurement to lattice and stack under A matrix
    iPos = int((yMeas/2+2)*5+xMeas/2+3)
    A = np.vstack((A,np.eye(nY*nX)[iPos-1,:]))

    #print("new Q:",Q)
    #print("Measured at (",xMeas,",",yMeas,")")
    #print("Value: ",zMeas)
    #print("new mean:",mu)

    ax2.contourf(xGT,yGT,mu.reshape(nY,nX))
    fig.canvas.draw()
    fig.canvas.flush_events()