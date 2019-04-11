# Parameter

# main.py

nIter = 700                     # number of iterations
ov2 = 0.01                      # measurement variance
dX = dY = 0.01                  # discretizaton in x and y
fastCalc = False                 # True: Fast Calculation, only one plot in the end; False: Live updating and plotting

(x0,y0) = (0,0)                 # initial state
maxStepsize = 0.5               # maximum change in every state per iteration

xMin = 0                        # GMRF dimensions
xMax = 10
nX = 10 
yMin = 0
yMax = 10
nY = 10
nBeta = 0                       # regression coeffiecients

# methods.py

# gmrfClass.py
valueT = 1e-3                   # Prior precision value for regression vector bet