# First 2D-GMRF Implementation

import numpy as np
import matplotlib.pyplot as pl
import math

x = y = np.arange(-2*math.pi,2*math.pi,0.01)
X, Y = np.meshgrid(x,y)

# Ground truth
f = lambda x,y: abs(np.sin(0.9*x)*np.sin(0.9*y))

# Plotting
fig = pl.figure()
pl.clf()
pl.contourf(X,Y,f(X,Y))
pl.show()