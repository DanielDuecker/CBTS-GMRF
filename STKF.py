import classes
import methods
import math
import scipy
import numpy as np

xMin = 0
xMax = 10
nX = 10
yMin = 0
yMax = 10
nY = 10
nBeta = 0

gmrf1 = classes.gmrf(xMin, xMax, nX, yMin, yMax, nY, nBeta)

sigmaT = 0.01
lambd = 1

# State representation of Sr
F = -1/sigmaT * np.ones((1,1))
H = math.sqrt(2*lambd / sigmaT) * np.ones((1,1))
G = np.ones((1,1))
simga2 = 0.01

# Kernels
Ks = methods.getPrecisionMatrix(gmrf1)
h = lambda tau: lambd * math.exp(-abs(tau)/sigmaT)

SigmaZero = scipy.linalg.solve_lyapunov(F, G*G.T)


