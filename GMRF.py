# First 1D-GMRF Implementation

from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as pl

# Ground truth
f = lambda x: abs(np.sin(0.9*x).flatten())
x = np.linspace(-10,0,100)

# Modeling unobserved locations X as X ~ N( E(X|Y) , )

# Plotting
fig = pl.figure()
pl.clf()
pl.plot(f(x),x)
pl.show()