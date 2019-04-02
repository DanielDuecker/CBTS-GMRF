# First 2D-GMRF Implementation

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import math
import time
import methods

# Parameters
nIter = 1000

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
mue = np.zeros((nY,nX))
Q = np.eye(nY+nX)                   #TO DO: Edit Q Matrix

# Discrete/Continuous-Mapping A
#A = np.eye(nY+nX)                  #NOT FINISHED

# Plotting initial belief
ax2 = fig.add_subplot(122)
ax2.contourf(xGT,yGT,mue.reshape(nY,nX))

plt.show()

#Update and plot field belief
for i in range(nIter):

    # Get measurement at random position
    xMeas = np.random.choice(xGT)
    yMeas = np.random.choise(yGT)
    zMeas = methods.getMeasurement(xMeas,yMeas,f,0.01)

    # Update on mue
    # Update on sigma

    ax2.contourf(xGT,yGT,mue.reshape(nY,nX))
    fig.canvas.draw()
    fig.canvas.flush_events()