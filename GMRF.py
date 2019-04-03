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
ax1 = fig.add_subplot(221)
ax1.contourf(x,y,f(x,y))
plt.title("True field")

##_____GMRF______##
# Initial GMRF
nY,nX = zGT.shape
mu = np.array([np.zeros((nY,nX)).flatten()]).T
QCond = np.eye(nY*nX,nY*nX)
Q = np.eye(nY*nX,nY*nX)+np.random.rand(nY*nX,nY*nX)                   #TO DO: Edit Q Matrix
#Q = np.eye(nY*nX)

# Plotting initial belief (mean and variance)
ax2 = fig.add_subplot(222)
ax2.contourf(xGT,yGT,mu.reshape(nY,nX))
plt.title("Mean of belief")

ax3 = fig.add_subplot(223)
ax3.contourf(xGT,yGT,np.diag(QCond).reshape(nY,nX))
plt.title("Variance of belief")

plt.show()

# Get first measurement:
xMeas = np.random.choice(xGT)
yMeas = np.random.choice(yGT)
zMeas = np.array([methods.getMeasurement(xMeas,yMeas,f,oz2)])
iPos = int((yMeas/2+2)*5+xMeas/2+3)
A = np.array([np.eye(nY*nX)[iPos,:]])

#Update and plot field belief
for i in range(nIter):
    print("Iteration ",i," of ",nIter,".")

    # Update mean
    temp1 = zMeas - np.dot(A,mu)
    temp2 = np.dot(A.T,temp1)
    mu = mu + 1/oz2*np.dot(np.linalg.inv(QCond),temp2)

    # Update precision matrix
    QCond = (Q+1/oz2*np.dot(A.T,A))

    # Get next measurement at random position, stack under measurement vector
    xMeas = np.random.choice(xGT)
    yMeas = np.random.choice(yGT)
    zMeas = np.vstack((zMeas,methods.getMeasurement(xMeas,yMeas,f,oz2)))
    
    # Map measurement to lattice and stack under A matrix
    iPos = int((yMeas/2+2)*5+xMeas/2+3)
    A = np.vstack((A,np.eye(nY*nX)[iPos-1,:]))

    # Plotting:
    ax2.contourf(xGT,yGT,mu.reshape(nY,nX))
    ax3.contourf(xGT,yGT,np.diag(QCond).reshape(nY,nX))
    fig.canvas.draw()
    fig.canvas.flush_events()
    print("New mean: \n",mu)