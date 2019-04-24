import classes
import methods
import math
import scipy
import numpy as np

T = 100
dt = 0.1
nIter = int(T/dt)

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

xHist = [par.x0]  # x-state history vector
yHist = [par.y0]  # y-state history vector


# State representation of Sr
F = -1/sigmaT * np.ones((1,1))
H = math.sqrt(2*lambd / sigmaT) * np.ones((1,1))
G = np.ones((1,1))
simga2 = 0.01

# Kernels
Ks = methods.getPrecisionMatrix(gmrf1)
KsChol = np.linalg.cholesky(Ks)
h = lambda tau: lambd * math.exp(-abs(tau)/sigmaT)

sigmaZero = scipy.linalg.solve_lyapunov(F, G*G.T)

# Initialization
sZero = 0
Cov = np.kron(np.eye(sigmaZero.shape[0]), sigmaT)

tk = 0

for i in range(nIter):
    t = i*dt
    A = scipy.linalg.expm(np.kron(np.eye(F.shape[0]), F) * (t - tk))

    (xMeas, yMeas) = methods.getNextState(xMeas, yMeas, xHist[-2], yHist[-2], par.maxStepsize, gmrf1)
    xHist.append(xMeas)
    yHist.append(yMeas)
    zMeas[(i+1) % par.nMeas] = methods.getMeasurement(xMeas, yMeas, trueField, par.ov2)

    if (t-tk)!=0:
        # Open loop prediciton
        sHead = A*skk
        sigmaS = np.dot(A,np.dot(Covkk,A.T))
    else:
        Phi = methods.mapConDis(gmrf1, xMeas, yMeas)
        C = np.dot(Phi,np.dot(KsChol,np.kron(eye(H.shape[0]), H)))
        QBar = scipy.integrate.quad(lambda tau: np.dot(scipy.linalg.expmnp.dot(F,tau),np.dot(G,np.dot(G.T,scipy.linalg.expmnp.dot(F,tau.T)))),0,dt)
        Q = np.kron(np.eye(QBar.shape[0]), QBar)
        R =sigma2

        #Kalman Regression

        tk = t
    f = 666
    sigmaF = 666



