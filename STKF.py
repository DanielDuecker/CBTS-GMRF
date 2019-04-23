import classes
import methods
import math
import scipy
import numpy as np

T = 100
dt = 0.1
nIter = T/dt

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

sigmaZero = scipy.linalg.solve_lyapunov(F, G*G.T)

# Initialization
sHeadZero = 0
sigma = np.kron(np.eye(sigmaZero.shape[0]), sigmaT)

tk = 0

for i in range(nIter):
    t = i*dt
    if t-tk =! 0:
        sHead = math.exp(np.kron(eye(F.shape[0]),F)*(t-tk))*skk
        sigmaS = 666
    else:
        A = 666
        C = 666
        Q = 666
        R = 666
        #Kalman Regression
        tk = t
    f = 666
    sigmaF = 666



