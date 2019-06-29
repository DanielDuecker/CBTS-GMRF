# classes for main.py

import copy

import math
import numpy as np
import scipy
from scipy import integrate
from scipy import interpolate

import methods
import parameters as par


class agent:
    def __init__(self, x0, y0, alpha0):
        self.x = x0
        self.y = y0
        self.alpha = alpha0  # angle of direction of movement
        self.maxStepsize = par.maxStepsize
        self.trajScaling = par.trajScaling
        self.derivX = self.trajScaling * math.cos(self.alpha)
        self.derivY = self.trajScaling * math.sin(self.alpha)

    def stateDynamics(self, x, y, alpha, u):
        alpha += u
        x += self.maxStepsize * math.cos(alpha)
        y += self.maxStepsize * math.sin(alpha)
        return x, y, alpha

    def trajectoryFromControl(self, u):
        xTraj = np.zeros((len(u), 1))
        yTraj = np.zeros((len(u), 1))
        alphaTraj = np.zeros((len(u), 1))
        (xTraj[0], yTraj[0], alphaTraj[0]) = (self.x, self.y, self.alpha)
        for i in range(len(u) - 1):
            (xTraj[i + 1], yTraj[i + 1], alphaTraj[i + 1]) = self.stateDynamics(xTraj[i], yTraj[i], alphaTraj[i], u[i])
        return xTraj, yTraj, alphaTraj


class trueField:
    def __init__(self, xEnd, yEnd, fieldType):
        self.fieldType = fieldType

        """parameters for temporal fields"""
        self.xShift = 0
        self.yShift = 0
        self.cScale = 1

        self.xRotationCenter = (par.xMax + par.xMin) / 2
        self.yRotationCenter = (par.yMax + par.yMin) / 2
        self.xPeak = (par.xMax + par.xMin) / 2
        self.yPeak = (par.yMax + par.yMin) * 3 / 4
        self.radius = math.sqrt((self.xPeak - self.xRotationCenter) ** 2 + (self.yPeak - self.yRotationCenter) ** 2)
        self.angleChange = -math.pi / 2
        self.peakValue = 15

        self.xEnd = xEnd
        self.yEnd = yEnd

        self.fInit, zGT = self.getFieldFunction()

        if self.fieldType == 'peak':
            self.fieldMin = np.min([-0.3, np.amin(zGT)])
            self.fieldMax = 1
        else:
            self.fieldMin = np.min([0, np.amin(zGT)])
            self.fieldMax = np.amax(zGT)
        self.fieldLevels = np.linspace(self.fieldMin, self.fieldMax, 20)

    def getFieldFunction(self):
        if self.fieldType == 'sine':
            xGT = np.arange(par.xMin, par.xMax, par.dX)
            yGT = np.arange(par.yMin, par.yMax, par.dY)
            XGT, YGT = np.meshgrid(xGT, yGT)
            zGT = np.sin(XGT) + np.sin(YGT)
            self.fInit = interpolate.interp2d(xGT, yGT, zGT)

        elif self.fieldType == 'peak':
            xGT = np.linspace(par.xMin, par.xMax, par.nGridX)
            yGT = np.linspace(par.yMin, par.yMax, par.nGridY)
            zGT = np.zeros((len(xGT), len(yGT)))
            for row in range(len(yGT)):
                for column in range(len(xGT)):
                    squaredDistance = np.linalg.norm(
                        np.array([[xGT[column]], [yGT[row]]]) - np.array([[self.xPeak], [self.yPeak]]), 2)
                    zGT[row, column] = np.exp(-.5 * 1 / 0.5 * squaredDistance)

        else:
            xGT = np.array([0, 2, 4, 6, 9])  # column coordinates
            yGT = np.array([0, 1, 3, 5, 9])  # row coordinates
            # zGT = np.array([[1, 2, 2, 1, 1],
            # [2, 4, 2, 1, 1],
            # [1, 2, 3, 3, 2],
            # [1, 1, 2, 3, 3],
            # [1, 1, 2, 3, 3]])

            zGT = np.array([[2, 4, 6, 7, 8],
                            [2.1, 5, 7, 11.25, 9.5],
                            [3, 5.6, 8.5, 17, 14.5],
                            [2.5, 5.4, 6.9, 9, 8],
                            [2, 2.3, 4, 6, 7.5]])

        return interpolate.interp2d(xGT, yGT, zGT), zGT

    def field(self, x, y):
        if self.fieldType == 'sine':
            return self.cScale * self.fInit(x, y)
        elif self.fieldType == 'peak':
            fField, zGT = self.getFieldFunction()
            return fField(x, y)
        elif self.fieldType == 'predefined':
            return self.fInit(x - self.xShift, y + self.yShift)

    def updateField(self, t):
        if t < par.pulseTime:
            self.xShift = par.dxdt * t % self.xEnd
            self.yShift = par.dydt * t % self.yEnd
            self.cScale = np.cos(math.pi * t / par.pulseTime)

            alpha = math.atan2((self.yPeak - self.yRotationCenter), (self.xPeak - self.xRotationCenter))
            self.xPeak = self.xRotationCenter + self.radius * math.cos(alpha + self.angleChange)
            self.yPeak = self.yRotationCenter + self.radius * math.sin(alpha + self.angleChange)


class GP:
    def __init__(self):
        self.kernelPar = par.kernelPar
        self.emptyData = True
        self.trainInput = None
        self.trainOutput = None

    def kernel(self, z1, z2):
        squaredDistance = np.linalg.norm(z1 - z2, 2)
        return np.exp(-.5 * 1 / self.kernelPar * squaredDistance)

    def getKernelMatrix(self, vec1, vec2):
        n = vec1.shape[0]
        N = vec2.shape[0]
        K = np.zeros((n, N))
        for i in range(n):
            for j in range(N):
                K[i, j] = self.kernel(vec1[i, :], vec2[j, :])
        return K

    def update(self, inputData, outputData):
        if self.emptyData:
            self.trainInput = np.expand_dims(inputData, axis=0)
            self.trainOutput = np.expand_dims(outputData, axis=0)
            self.emptyData = False
        else:
            self.trainInput = np.vstack((self.trainInput, inputData))
            self.trainOutput = np.vstack((self.trainOutput, outputData))

    def predict(self, inputData):
        # according to https://www.cs.ubc.ca/~nando/540-2013/lectures/l6.pdf
        K = self.getKernelMatrix(self.trainInput, self.trainInput)
        L = np.linalg.cholesky(K)

        # Compute mean
        Lk = np.linalg.solve(L, self.getKernelMatrix(self.trainInput, inputData))
        mu = np.dot(Lk.T, np.linalg.solve(L, self.trainOutput))

        # Compute variance
        KStar = self.getKernelMatrix(inputData, inputData)
        var = KStar - np.dot(Lk.T, Lk)

        return mu, var


class gmrf:
    def __init__(self):
        """GMRF properties"""
        self.xMin = par.xMin
        self.xMax = par.xMax

        self.yMin = par.yMin
        self.yMax = par.yMax

        self.nEdge = par.nEdge

        self.ov2 = par.ov2
        self.valueT = par.valueT
        self.dt = par.dt

        # Distance between two vertices in x and y without edges
        self.dx = (self.xMax - self.xMin) / (par.nGridX - 1)
        self.dy = (self.yMax - self.yMin) / (par.nGridY - 1)

        self.nY = par.nGridY + 2 * self.nEdge  # Total number of vertices in y with edges
        self.nX = par.nGridX + 2 * self.nEdge  # Total number of vertices in x with edges
        self.nP = self.nX * self.nY  # Total number of vertices

        self.xMinEdge = self.xMin - self.nEdge * self.dx
        self.xMaxEdge = self.xMax + self.nEdge * self.dx
        self.yMinEdge = self.yMin - self.nEdge * self.dy
        self.yMaxEdge = self.yMax + self.nEdge * self.dy

        self.x = np.linspace(self.xMinEdge, self.xMaxEdge, self.nX)  # Vector of x grid values
        self.y = np.linspace(self.yMinEdge, self.yMaxEdge, self.nY)  # Vector of y grid values

        "Mean augmented bayesian regression"
        # Mean regression matrix
        self.nBeta = par.nBeta
        F = np.ones((self.nP, self.nBeta))
        self.Tinv = np.linalg.inv(self.valueT * np.eye(self.nBeta))

        # Precision matrix for z values (without regression variable beta)
        self.Lambda = methods.getPrecisionMatrix(self)

        # Augmented prior covariance matrix
        covPriorUpperLeft = np.linalg.inv(self.Lambda) + np.dot(F, np.dot(self.Tinv, F.T))
        covPriorUpperRight = np.dot(F, self.Tinv)
        covPriorLowerLeft = np.dot(F, self.Tinv).T
        covPriorLowerRight = self.Tinv
        self.covPrior = np.vstack(
            (np.hstack((covPriorUpperLeft, covPriorUpperRight)), np.hstack((covPriorLowerLeft, covPriorLowerRight))))
        self.meanPrior = np.zeros((self.nP + self.nBeta, 1))

        # Initialize augmented conditioned mean, covariance and precision matrices
        self.meanCond = np.zeros((self.nP + self.nBeta, 1))
        self.covCond = self.covPrior
        self.diagCovCond = self.covCond.diagonal().reshape(self.nP + self.nBeta, 1)
        self.precCond = np.linalg.inv(self.covCond)
        self.covLevels = np.linspace(0, min(np.amax(self.diagCovCond), 0.9), 20)  # using np.amax(self.diagCovCond)
        # leads to wrong scaling, since self.diagCovCond is initialized too hight due to T_inv

        "Sequential bayesian regression"
        self.bSeq = np.zeros((self.nP + self.nBeta, 1))

    def bayesianUpdate(self, zMeas, Phi):
        """Update conditioned precision matrix"""
        R = np.dot(Phi, np.dot(self.covPrior, Phi.T)) + self.ov2 * np.eye(
            len(zMeas))  # covariance matrix of measurements
        temp1 = np.dot(Phi, self.covPrior)
        temp2 = np.dot(np.linalg.inv(R), temp1)
        temp3 = np.dot(Phi.T, temp2)

        self.covCond = self.covPrior - np.dot(self.covPrior, temp3)
        # self.covCond = np.linalg.inv((np.linalg.inv(self.covPrior)+1/self.ov2*np.dot(Phi.T,Phi))) # alternative way
        self.diagCovCond = self.covCond.diagonal().reshape(self.nP + self.nBeta, 1)

        "Update mean"
        if par.truncation:
            self.meanCond = self.meanPrior + 1 / self.ov2 * np.dot(self.covCond,
                                                                   np.dot(Phi.T, zMeas - np.dot(Phi, self.meanPrior)))
        else:
            self.meanCond = np.dot(self.covPrior, np.dot(Phi.T, np.dot(np.linalg.inv(R), zMeas)))

    def seqBayesianUpdate(self, zMeas_k, Phi_k):
        Phi_k = Phi_k.reshape(1, self.nP + self.nBeta)
        zMeas_k = zMeas_k.reshape(1, 1)

        self.bSeq = self.bSeq + 1 / self.ov2 * Phi_k.T * zMeas_k  # sequential update canonical mean
        self.precCond = self.precCond + 1 / self.ov2 * np.dot(Phi_k.T, Phi_k)  # sequential update of precision matrix
        self.covCond = np.linalg.inv(self.precCond)
        self.diagCovCond = self.covCond.diagonal().reshape(self.nP + self.nBeta, 1)  # works too
        self.meanCond = np.dot(np.linalg.inv(self.precCond), self.bSeq)

        # TODO: Fix calculation of covariance diagonal
        # hSeq = np.linalg.solve(self.precCond, Phi_k.T)
        # self.diagCovCond = self.diagCovCond - 1 / (self.ov2 + np.dot(Phi_k, hSeq)[0, 0]) * np.dot(hSeq,
        #                                                            hSeq.T).diagonal().reshape(self.nP + self.nBeta, 1)


class stkf:
    def __init__(self, gmrf1):
        self.gmrf = gmrf1
        self.dt = par.dt
        self.sigmaT = par.sigmaT
        self.lambdSTKF = par.lambdSTKF

        # State representation of Sr
        self.F = -1 / self.sigmaT * np.eye(1)
        self.H = math.sqrt(2 * self.lambdSTKF / self.sigmaT) * np.eye(1)
        self.G = np.eye(1)
        self.sigma2 = par.sigma2

        # Kernels
        self.Ks = np.linalg.inv(methods.getPrecisionMatrix(self.gmrf))
        self.KsChol = np.linalg.cholesky(self.Ks)
        # h = lambda tau: lambdSTKF * math.exp(-abs(tau) / sigmaT) # used time kernel

        self.sigmaZero = scipy.linalg.solve_continuous_lyapunov(self.F, -self.G * self.G.T)

        self.A = scipy.linalg.expm(np.kron(np.eye(self.gmrf.nP), self.F) * self.dt)
        self.Cs = np.dot(self.KsChol, np.kron(np.eye(self.gmrf.nP), self.H))
        QBar = scipy.integrate.quad(lambda tau: np.dot(scipy.linalg.expm(np.dot(self.F, tau)), np.dot(self.G,
                                            np.dot(self.G.T,scipy.linalg.expm(np.dot(self.F,tau)).T))),0, self.dt)[0]
        self.Q = np.kron(np.eye(self.gmrf.nP), QBar)
        self.R = self.sigma2 * np.eye(1)

        # Initialization
        self.skk = np.zeros((self.gmrf.nP, 1))
        self.covkk = np.kron(np.eye(self.gmrf.nP), self.sigmaZero)

    def kalmanFilter(self, t, xMeas, yMeas, zMeas):
        if t % 1 != 0:
            # Open loop prediciton
            st = np.dot(self.A, self.skk)
            covt = np.dot(self.A, np.dot(self.covkk, self.A.T))
        else:
            Phi = methods.mapConDis(self.gmrf, xMeas, yMeas)
            C = np.dot(Phi, self.Cs)

            # Kalman Regression
            sPred = np.dot(self.A, self.skk)
            covPred = np.dot(self.A, np.dot(self.covkk, self.A.T)) + self.Q
            kalmanGain = np.dot(covPred, np.dot(C.T, np.linalg.inv(np.dot(C, np.dot(covPred, C.T)) + self.R)))
            sUpdated = sPred + np.dot(kalmanGain, zMeas - np.dot(C, sPred))
            covUpdated = np.dot(np.eye(self.gmrf.nP) - np.dot(kalmanGain, C), covPred)
            self.skk = sUpdated
            self.covkk = covUpdated

            st = sUpdated
            covt = covUpdated

        self.gmrf.meanCond = np.dot(self.Cs, st)
        self.gmrf.covCond = np.dot(self.Cs, np.dot(covt, self.Cs.T))
        self.gmrf.diagCovCond = self.gmrf.covCond.diagonal()


class node:
    def __init__(self, gmrf1, auv):
        self.gmrf = copy.deepcopy(gmrf1)
        self.auv = copy.deepcopy(auv)
        self.rewardToNode = 0
        self.accReward = 0
        self.actionToNode = []
        self.depth = 0
        self.parent = []
        self.children = []
        self.visits = 1
        self.D = []
        self.GP = GP()
