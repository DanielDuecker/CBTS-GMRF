# First 2D-GMRF Implementation

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import math
import time

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

# Show ground thruth interpolation points
XGT,YGT = np.meshgrid(xGT,yGT)
ax1.plot(XGT,YGT,marker='.',color='k',linestyle='none')

g = lambda x,y: abs(np.sin(0.9*x)*np.sin(0.9*y))
ax2 = fig.add_subplot(122)
ax2.contourf(X,Y,g(X,Y))

plt.show()

#Update and plot field belief
for i in range(nIter):
    g = lambda x,y: abs(np.sin(0.9*x)*np.sin(0.9*y-i/10))
    ax2.contourf(X,Y,g(X,Y))
    fig.canvas.draw()
    fig.canvas.flush_events()