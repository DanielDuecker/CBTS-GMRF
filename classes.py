# Parameter file for main.py

import numpy as np

import methods
import math
import parameters as par

from scipy import interpolate
from scipy import  integrate
import scipy

class gmrf:
    def __init__(self, xMin, xMax, nX, yMin, yMax, nY, nBeta):
        """GMRF properties"""
        self.xMin = xMin
        self.xMax = xMax
        self.nX = nX  # Total number of vertices in x
        self.x = np.linspace(self.xMin, self.xMax, self.nX)  # Vector of x grid values

        self.yMin = yMin
        self.yMax = yMax
        self.nY = nY  # Total number of vertices in y
        self.y = np.linspace(self.yMin, self.yMax, self.nY)  # Vector of y grid values

        self.nP = nX * nY  # Total number of vertices

        # Distance between two vertices in x and y
        self.dx = (self.xMax - self.xMin) / (self.nX - 1)
        self.dy = (self.yMax - self.yMin) / (self.nY - 1)

        "Mean augmented bayesian regression"
        # Mean regression matrix
        F = np.ones((self.nP, nBeta))
        self.nBeta = nBeta
        self.Tinv = np.linalg.inv(par.valueT * np.eye(self.nBeta))

        # Precision matrix for z values (without regression variable beta)
        self.Lambda = methods.getPrecisionMatrix(self)

        # Augmented prior covariance matrix
        covPriorUpperLeft = np.linalg.inv(self.Lambda) + np.dot(F, np.dot(self.Tinv, F.T))
        covPriorUpperRight = np.dot(F, self.Tinv)
        covPriorLowerLeft = np.dot(F, self.Tinv).T
        covPriorLowerRight = self.Tinv
        self.covPrior = np.vstack(
            (np.hstack((covPriorUpperLeft, covPriorUpperRight)), np.hstack((covPriorLowerLeft, covPriorLowerRight))))
        self.meanPrior = np.zeros((self.nP+self.nBeta, 1))

        # Initialize augmented conditioned mean, covariance and precision matrices
        self.meanCond = np.zeros((self.nP + self.nBeta, 1))
        self.covCond = self.covPrior
        self.diagCovCond = self.covCond.diagonal().reshape(self.nP + self.nBeta, 1)
        self.precCond = np.linalg.inv(self.covCond)

        "Sequential bayesian regression"
        self.bSeq = np.zeros((self.nP + self.nBeta, 1))

    def bayesianUpdate(self, zMeas, Phi):
        """Update conditioned precision matrix"""
        R = np.dot(Phi, np.dot(self.covPrior, Phi.T)) + par.ov2 * np.eye(
            len(zMeas))  # covariance matrix of measurements
        temp1 = np.dot(Phi, self.covPrior)
        temp2 = np.dot(np.linalg.inv(R), temp1)
        temp3 = np.dot(Phi.T, temp2)
        self.covCond = self.covPrior - np.dot(self.covPrior, temp3)
        # self.covCond = np.linalg.inv((np.linalg.inv(self.covPrior)+1/par.ov2*np.dot(Phi.T,Phi))) # alternative way
        self.diagCovCond = self.covCond.diagonal().reshape(self.nP + self.nBeta, 1)

        "Update mean"
        if par.truncation:
            self.meanCond = self.meanPrior + 1 / par.ov2 * np.dot(self.covCond,
                                                                  np.dot(Phi.T, zMeas - np.dot(Phi, self.meanPrior)))
        else:
            self.meanCond = np.dot(self.covPrior, np.dot(Phi.T, np.dot(np.linalg.inv(R), zMeas)))

    def seqBayesianUpdate(self, zMeas_k, Phi_k):
        Phi_k = Phi_k.reshape(1, self.nP + self.nBeta)
        zMeas_k = zMeas_k.reshape(1, 1)

        self.bSeq = self.bSeq + 1 / par.ov2 * Phi_k.T * zMeas_k  # sequential update canonical mean
        self.precCond = self.precCond + 1 / par.ov2 * np.dot(Phi_k.T, Phi_k)  # sequential update of precision matrix
        self.diagCovCond = np.linalg.inv(self.precCond).diagonal().reshape(self.nP+self.nBeta, 1)  # works too
        self.meanCond = np.dot(np.linalg.inv(self.precCond), self.bSeq)

        # TODO: Fix calculation of covariance diagonal
        #hSeq = np.linalg.solve(self.precCond, Phi_k.T)
        #self.diagCovCond = self.diagCovCond - 1 / (par.ov2 + np.dot(Phi_k, hSeq)[0, 0]) * np.dot(hSeq,
        #                                                            hSeq.T).diagonal().reshape(self.nP + self.nBeta, 1)

class trueField:
    def __init__(self, xEnd, yEnd, sinusoidal, temporal):
        self.sinusoidal = sinusoidal
        self.temporal = temporal

        self.xShift = 0
        self.yShift = 0
        self.cScale = 1

        self.xEnd = xEnd
        self.yEnd = yEnd

        if self.sinusoidal:
            xGT = np.arange(par.xMin, par.xMax, par.dX)
            yGT = np.arange(par.yMin, par.yMax, par.dY)
            XGT, YGT = np.meshgrid(xGT,yGT)
            zGT = np.sin(XGT)+np.sin(YGT)
            self.fInit = interpolate.interp2d(xGT, yGT, zGT)
        else:
            xGT = np.array([0, 2, 4, 6, 9])  # column coordinates
            yGT = np.array([0, 1, 3, 5, 9])  # row coordinates
            zGT = np.array([[1, 2, 2, 1, 1],
                            [2, 4, 2, 1, 1],
                            [1, 2, 3, 3, 2],
                            [1, 1, 2, 3, 3],
                            [1, 1, 2, 3, 3]])

        self.fInit = interpolate.interp2d(xGT, yGT, zGT)

    def field(self, x, y):
        if not par.sinusoidal:
            return self.fInit(x-self.xShift, y+self.yShift)
        else:
            return self.cScale*self.fInit(x, y)

    def updateField(self, t):
        self.xShift = par.dxdt*t % self.xEnd
        self.yShift = par.dydt*t % self.yEnd
        if t < par.pulseTime:
            self.cScale = np.cos(math.pi*t/par.pulseTime)

class stkf:
    def __init__(self,xMin, xMax, nX, yMin, yMax, nY, nBeta,trueField,dt,sigmaT,lambd,sigma2):
        self.gmrf = gmrf(xMin, xMax, nX, yMin, yMax, nY, nBeta)
        self.trueField = trueField
        self.dt = dt
        self.sigmaT = sigmaT
        self.lambd = lambd

        # State representation of Sr
        self.F = -1 / sigmaT * np.ones((1, 1))
        self.H = math.sqrt(2 * lambd / sigmaT) * np.ones((1, 1))
        self.G = np.ones((1, 1))
        self.sigma2 = sigma2

        # Kernels
        self.Ks = methods.getPrecisionMatrix(self.gmrf)
        self.KsChol = np.linalg.cholesky(self.Ks)
        # h = lambda tau: lambd * math.exp(-abs(tau) / sigmaT)

        self.sigmaZero = scipy.linalg.solve_lyapunov(self.F, self.G * self.G.T)

        # Initialization
        self.skk = np.zeros((self.gmrf.nP, 1))
        self.covkk = np.kron(np.eye(self.gmrf.nP), self.sigmaZero)
        self.tk = 0

    def kalmanFilter(self,t,xMeas,yMeas,zMeas):
            A = scipy.linalg.expm(np.kron(np.eye(self.gmrf.nP), self.F) * (t - self.tk))
            if t % 1 != 0:
                # Open loop prediciton
                st = np.dot(A, self.skk)
                covt = np.dot(A, np.dot(self.covkk, A.T))
            else:
                Phi = methods.mapConDis(self.gmrf, xMeas, yMeas)
                C = np.dot(Phi, np.dot(self.KsChol, np.kron(np.eye(self.gmrf.nP), self.H)))
                QBar = scipy.integrate.quad(lambda tau: np.dot(scipy.linalg.expm(np.dot(self.F, tau)), np.dot(self.G, np.dot(self.G.T, scipy.linalg.expm(np.dot(self.F,tau).T)))),0, self.dt)[0]
                Q = np.kron(np.eye(self.gmrf.nP), QBar)
                R = self.sigma2

                # Kalman Regression
                sPred = np.dot(A, self.skk)
                covPred = np.dot(A, np.dot(self.covkk, A.T)) + Q

                kalmanGain = np.dot(covPred, np.dot(C.T, np.linalg.inv(np.dot(C, np.dot(covPred, C.T)) + R)))
                sUpdated = sPred + np.dot(kalmanGain, zMeas - np.dot(C, sPred))
                covUpdated = np.dot(np.eye(self.gmrf.nP) - np.dot(kalmanGain, C), covPred)
                self.skk = sUpdated
                self.covkk = covUpdated

                st = sUpdated
                covt = covUpdated

            hAug = np.kron(np.eye(self.gmrf.nP), self.H)
            self.gmrf.meanCond = np.dot(self.KsChol, np.dot(hAug, st))
            self.gmrf.covCond = np.dot(self.KsChol, np.dot(hAug, np.dot(covt, np.dot(hAug.T, self.KsChol))))
            self.gmrf.diagCovCond = self.gmrf.covCond.diagonal()