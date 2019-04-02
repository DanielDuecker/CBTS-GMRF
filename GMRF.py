# First 2D-GMRF Implementation

import numpy as np
import matplotlib.pyplot as pl
import math
import time

# Parameters
nIter = 1000

# Set up axis
x = y = np.arange(-2*math.pi,2*math.pi,0.01)
X, Y = np.meshgrid(x,y)

# Ground truth
f = lambda x,y: abs(np.sin(0.9*x)*np.sin(0.9*y))

# Plotting ground truth
pl.ion()
fig = pl.figure()

ax1 = fig.add_subplot(121)
ax1.contourf(X,Y,f(X,Y))

g = lambda x,y: abs(np.sin(0.9*x)*np.sin(0.9*y))
ax2 = fig.add_subplot(122)
ax2.contourf(X,Y,g(X,Y))

pl.show()

#Update and plot field belief
for i in range(nIter):
    g = lambda x,y: abs(np.sin(0.9*x)*np.sin(0.9*y-i/10))
    ax2.contourf(X,Y,g(X,Y))
    fig.canvas.draw()
    fig.canvas.flush_events()

