import classes
import methods
import math
import scipy

xMin = 0
xMax = 10
nX = 10
yMin = 0
yMax = 10
nY = 10

gmrf1 = classes.gmrf(xMin,xMax,nX,yMin,yMax,nY)

sigmaT = 0.01
lambd = 1

# State representation of Sr
F = -1/sigmaT
H = math.sqrt(2*lambd / sigmaT)
G = 1
simga2 = 0.01

# Kernels
Ks = methods.getPrecisionMatrix(gmrf1)
h = lambda tau: lambd * math.exp(-abs(tau)/sigmaT)

SigmaZero = scipy.linalg.solve_lyapunov(F, G*G.T)


