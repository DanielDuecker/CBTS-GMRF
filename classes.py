# classes for main.py

import copy

import math
import numpy as np
import scipy
from scipy import integrate
from scipy import interpolate
import scipy.sparse as sp

import functions


class agent:
    def __init__(self,par, x0, y0, alpha0):
        self.x = x0
        self.y = y0
        self.alpha = alpha0  # angle of direction of movement
        self.maxStepsize = par.maxStepsize
        self.trajScaling = par.trajScaling
        self.derivX = self.trajScaling * math.cos(self.alpha)
        self.derivY = self.trajScaling * math.sin(self.alpha)

    def stateDynamics(self, x, y, alpha, u):
        alphaNext = alpha + u
        xNext = x + self.maxStepsize * math.cos(alphaNext)
        yNext = y + self.maxStepsize * math.sin(alphaNext)

        if alphaNext > 2 * math.pi:
            a_pi = int(alphaNext / (2 * math.pi))
            alphaNext = alphaNext - a_pi * 2 * math.pi

        if alphaNext < 0:
            a_pi = int(- alphaNext / (2 * math.pi))+1
            alphaNext = alphaNext + a_pi * 2 * math.pi

        return xNext, yNext, alphaNext

    def trajectoryFromControl(self, u):
        xTraj = np.zeros((len(u), 1))
        yTraj = np.zeros((len(u), 1))
        alphaTraj = np.zeros((len(u), 1))
        (xTraj[0], yTraj[0], alphaTraj[0]) = (self.x, self.y, self.alpha)
        for i in range(len(u) - 1):
            (xTraj[i + 1], yTraj[i + 1], alphaTraj[i + 1]) = self.stateDynamics(xTraj[i], yTraj[i], alphaTraj[i], u[i])
        return xTraj, yTraj, alphaTraj


class trueField:
    def __init__(self, par, fieldType):
        self.fieldType = fieldType

        """random field"""
        maxRandomValue = 5
        xRand = np.linspace(0,10,11)  # column coordinates
        yRand = np.linspace(0,10,11)  # row coordinates
        zRand = np.random.rand(11,11)*maxRandomValue
        self.fRand = interpolate.interp2d(xRand, yRand, zRand)

        """sine field"""
        self.cScale = 1

        """peak field"""
        self.xRotationCenter = (par.xMax + par.xMin) / 2
        self.yRotationCenter = (par.yMax + par.yMin) / 2
        self.xPeak = (par.xMax + par.xMin) / 2
        self.yPeak = (par.yMax + par.yMin) * 3 / 4
        self.radius = math.sqrt((self.xPeak - self.xRotationCenter) ** 2 + (self.yPeak - self.yRotationCenter) ** 2)
        self.angleChange = -math.pi / 8
        self.peakValue = 10
        self.peakPar = 0.8

        """predefined field"""
        self.xShift = 0
        self.yShift = 0
        xPreDef = np.array([0, 2, 4, 6, 9])  # column coordinates
        yPreDef = np.array([0, 1, 3, 5, 9])  # row coordinates
        zPreDef = np.array([[1, 2, 2, 1, 1],
        [2, 4, 2, 1, 1],
        [1, 2, 3, 3, 2],
        [1, 1, 2, 3, 3],
        [1, 1, 2, 3, 3]])
        #zPreDef = np.array([[2, 4, 6, 7, 8],
        #              [2.1, 5, 7, 11.25, 9.5],
        #              [3, 5.6, 8.5, 17, 14.5],
        #              [2.5, 5.4, 6.9, 9, 8],
        #              [2, 2.3, 4, 6, 7.5]])
        self.minValPreDef = np.min((0,np.min(zPreDef)-1))
        self.maxValPreDef = np.max(zPreDef)+1
        self.fPreDef = interpolate.interp2d(xPreDef, yPreDef, zPreDef)

        if self.fieldType == 'sine':
            self.fieldMin = -2.5
            self.fieldMax = 2.5
        elif self.fieldType == 'peak':
            self.fieldMin = -1
            self.fieldMax = self.peakValue + 0.1
        elif self.fieldType == 'random':
            self.fieldMin = -maxRandomValue
            self.fieldMax = maxRandomValue
        else:
            self.fieldMin = self.minValPreDef-par.ov2
            self.fieldMax = self.maxValPreDef+par.ov2

        self.fieldLevels = np.linspace(self.fieldMin, self.fieldMax, 20)

    def getField(self,x,y):
        if self.fieldType == 'sine':
            X,Y = np.meshgrid(x, y)
            Z = self.cScale*(np.sin(X) + np.sin(Y))
        elif self.fieldType == 'peak':
            X,Y = np.meshgrid(x, y)
            Z = self.peakValue * np.exp(-((X-self.xPeak)/self.peakPar)**2)*np.exp(-((Y-self.yPeak)/self.peakPar)**2)
        elif self.fieldType == 'random':
            Z = self.fRand(x,y)
        else:
            Z = self.fPreDef(x - self.xShift,y + self.yShift)
        return Z

    def updateField(self, par, t):
        if t < par.pulseTime:
            self.cScale = np.cos(10*math.pi * t / par.pulseTime)

            alpha = math.atan2((self.yPeak - self.yRotationCenter), (self.xPeak - self.xRotationCenter))
            self.xPeak = self.xRotationCenter + self.radius * math.cos(alpha + self.angleChange)
            self.yPeak = self.yRotationCenter + self.radius * math.sin(alpha + self.angleChange)

            self.xShift = par.dxdt * t
            self.yShift = par.dydt * t


class GP:
    def __init__(self, par):
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
    def __init__(self,par,nGridX,nGridY,nEdge):
        "GMRF properties"
        self.par = par

        self.xMin = par.xMin
        self.xMax = par.xMax
        self.yMin = par.yMin
        self.yMax = par.yMax

        self.nGridX = nGridX
        self.nGridY = nGridY
        self.nEdge = nEdge

        self.ov2 = par.ov2
        self.valueT = par.valueT # Precision value for beta regression
        self.dt = par.dt

        "Distance between two vertices in x and y without edges"
        self.dx = round((self.xMax - self.xMin) / (self.nGridX - 1),5)
        self.dy = round((self.yMax - self.yMin) / (self.nGridY - 1),5)

        self.nY = self.nGridY + 2 * self.nEdge  # Total number of vertices in y with edges
        self.nX = self.nGridX + 2 * self.nEdge  # Total number of vertices in x with edges
        self.nP = self.nX * self.nY  # Total number of vertices

        self.xMinEdge = self.xMin - self.nEdge * self.dx
        self.xMaxEdge = self.xMax + self.nEdge * self.dx
        self.yMinEdge = self.yMin - self.nEdge * self.dy
        self.yMaxEdge = self.yMax + self.nEdge * self.dy

        self.x = np.linspace(self.xMinEdge, self.xMaxEdge, self.nX)  # Vector of x grid values
        self.y = np.linspace(self.yMinEdge, self.yMaxEdge, self.nY)  # Vector of y grid values

        "Precision matrix for z values (without regression variable beta)"
        self.Lambda = functions.getPrecisionMatrix(self)

        "Mean augmented bayesian regression"
        self.nBeta = par.nBeta

        F = np.ones((self.nP, self.nBeta))
        FSparse = sp.csr_matrix(F)
        FTSparse = sp.csr_matrix(F.T)
        T = self.valueT * np.eye(self.nBeta)
        Tinv = np.linalg.inv(T)
        TSparse = sp.csr_matrix(T)
        TinvSparse = sp.csr_matrix(Tinv)

        "Augmented prior precision matrix"
        precPriorUpperRight = self.Lambda.dot(-1 * FSparse)
        precPriorLowerLeft = -1 * FTSparse.dot(self.Lambda)
        precPriorLowerRight = sp.csr_matrix.dot(FTSparse, self.Lambda.dot(FSparse)) + TSparse
        precH1 = sp.hstack([self.Lambda, precPriorUpperRight])
        precH2 = sp.hstack([precPriorLowerLeft, precPriorLowerRight])
        self.precCondSparse = sp.vstack([precH1, precH2]).tocsr()

        "Augmented prior covariance matrix"
        covPriorUpperLeft = sp.linalg.inv(self.Lambda.tocsc()) + sp.csr_matrix.dot(FSparse,TinvSparse.dot(FTSparse))
        covPriorUpperRight = FSparse.dot(Tinv)
        covPriorLowerLeft = covPriorUpperRight.T
        covPriorLowerRight = TinvSparse
        covH1 = sp.hstack([covPriorUpperLeft, covPriorUpperRight])
        covH2 = sp.hstack([covPriorLowerLeft, covPriorLowerRight])
        self.covPrior = np.array(sp.vstack([covH1, covH2]).todense())
        self.covCond = self.covPrior

        self.diagCovCond = self.covCond.diagonal().reshape(self.nP + self.nBeta, 1)

        "Prior and conditioned mean"
        self.meanPrior = np.zeros((self.nP + self.nBeta, 1))
        self.meanCond = np.zeros((self.nP + self.nBeta, 1))

        self.covLevels = np.linspace(-0.2, min(np.amax(self.diagCovCond), 0.9), 20) # TODO Adapt

        "Sequential bayesian regression"
        self.bSeq = np.zeros((self.nP + self.nBeta, 1))


    def bayesianUpdate(self, fMeas, Phi):
        """Update conditioned precision matrix"""
        R = np.dot(Phi, np.dot(self.covPrior, Phi.T)) + self.ov2 * np.eye(
            len(fMeas))  # covariance matrix of measurements
        temp1 = np.dot(Phi, self.covPrior)
        temp2 = np.dot(np.linalg.inv(R), temp1)
        temp3 = np.dot(Phi.T, temp2)

        self.covCond = self.covPrior - np.dot(self.covPrior, temp3)
        # self.covCond = np.linalg.inv((np.linalg.inv(self.covPrior)+1/self.ov2*np.dot(Phi.T,Phi))) # alternative way
        self.diagCovCond = self.covCond.diagonal().reshape(self.nP + self.nBeta, 1)

        "Update mean"
        if self.par.belief == 'regBayesTrunc':
            self.meanCond = self.meanPrior + 1 / self.ov2 * np.dot(self.covCond,
                                                                   np.dot(Phi.T, fMeas - np.dot(Phi, self.meanPrior)))
        else:
            self.meanCond = np.dot(self.covPrior, np.dot(Phi.T, np.dot(np.linalg.inv(R), fMeas)))

        # Also update bSeq and precCond in case seq. belief update is used for planning
        PhiT = Phi.T
        PhiTSparse = sp.csr_matrix(PhiT)
        self.bSeq = self.bSeq + 1 / self.ov2 * np.dot(PhiT,fMeas)  # sequential update canonical mean
        self.precCondSparse = self.precCondSparse + 1 / self.par.ov2 * PhiTSparse.dot(PhiTSparse.T)


    def seqBayesianUpdate(self, fMeas, Phi):
        PhiT = Phi.T
        PhiTSparse = sp.csr_matrix(PhiT)

        hSeq = sp.linalg.spsolve(self.precCondSparse, PhiTSparse).T
        self.bSeq = self.bSeq + fMeas[0]/self.ov2 * PhiT  # sequential update canonical mean
        self.precCondSparse = self.precCondSparse + 1 / self.ov2 * PhiTSparse.dot(PhiTSparse.T)  # sequential update of precision matrix
        self.meanCond = sp.linalg.spsolve(self.precCondSparse, self.bSeq).reshape(self.nP+self.nBeta,1)
        self.diagCovCond = np.subtract(self.diagCovCond,np.multiply(hSeq,hSeq).reshape(self.nP+self.nBeta,1) / (self.ov2 + np.dot(Phi, hSeq)[0]))
        """ Works too:
        self.covCond = np.linalg.inv(self.precCond)
        self.diagCovCond = self.covCond.diagonal().reshape(self.nP + self.nBeta, 1)
        """

class stkf:
    def __init__(self,par, gmrf1):
        self.par = par
        self.gmrf = gmrf1
        self.dt = par.dt
        self.sigmaT = par.sigmaT
        self.lambdSTKF = par.lambdSTKF

        # State representation of Sr
        F = -1 / self.sigmaT * np.eye(1)
        H = math.sqrt(2 * self.lambdSTKF / self.sigmaT) * np.eye(1)
        G = np.eye(1)
        sigma2 = par.sigma2

        # Kernels
        Ks = np.linalg.inv(np.array(functions.getPrecisionMatrix(self.gmrf).todense()))
        KsChol = np.linalg.cholesky(Ks)
        # h = lambda tau: lambdSTKF * math.exp(-abs(tau) / sigmaT) # used time kernel

        sigmaZero = scipy.linalg.solve_continuous_lyapunov(F, -G * G.T)

        self.A = sp.csr_matrix(sp.linalg.expm(np.kron(np.eye(self.gmrf.nP), F) * self.dt))
        self.AT = sp.csr_matrix(self.A.T)
        self.CsDense = np.dot(KsChol, np.kron(np.eye(self.gmrf.nP), H))
        self.Cs = sp.csr_matrix(self.CsDense)
        QBar = scipy.integrate.quad(lambda tau: np.dot(scipy.linalg.expm(np.dot(F, tau)), np.dot(G,
                                            np.dot(G.T,scipy.linalg.expm(np.dot(F,tau)).T))),0, self.dt)[0]
        self.Q = sp.csr_matrix( np.kron(np.eye(self.gmrf.nP), QBar))
        self.R = sp.csr_matrix(sigma2 * np.eye(1))

        # Initialization
        self.skk = sp.csr_matrix(np.zeros((self.gmrf.nP, 1)))
        self.covkk = sp.csr_matrix(np.kron(np.eye(self.gmrf.nP), sigmaZero))

    def kalmanFilter(self, t, xMeas, yMeas, fMeas):
        import cProfile
        if t % 1 != 0:
            # Open loop prediciton
            self.skk = np.dot(self.A, self.skk)
            self.covkk = np.dot(self.A, np.dot(self.covkk, self.A.T))
        else:
            Phi = functions.mapConDis(self.gmrf, xMeas, yMeas)
            PhiSparse = sp.csr_matrix(Phi)
            C = PhiSparse.dot(self.Cs)
            CT = sp.csr_matrix(C.T)

            # Kalman Regression
            sPred = self.A.dot(self.skk)
            covPred = self.A.dot(self.covkk.dot(self.AT)) + self.Q
            denum = sp.csr_matrix(sp.linalg.inv(C.dot(covPred.dot(CT)) + self.R))
            kalmanGain = covPred.dot(CT.dot(denum))
            self.skk = sPred + kalmanGain.dot(sp.csr_matrix(fMeas) - C.dot(sPred))
            self.covkk = (sp.csr_matrix(np.eye(self.gmrf.nP)) - kalmanGain.dot(C)).dot(covPred)

        self.gmrf.meanCond = np.array(np.dot(self.CsDense,self.skk.todense()),ndmin=2)
        self.gmrf.covCond = np.array(np.dot(self.CsDense,np.dot(self.covkk.todense(),self.CsDense.T)))
        self.gmrf.diagCovCond = self.gmrf.covCond.diagonal().reshape(self.gmrf.nP+self.gmrf.nBeta,1)

        # Also update bSeq and precCond in case seq. belief update is used for planning
        PhiT = Phi.T
        PhiTSparse = sp.csr_matrix(PhiT)
        self.gmrf.bSeq = self.gmrf.bSeq + 1 / self.par.ov2 * PhiT * fMeas  # sequential update canonical mean
        self.gmrf.precCondSparse = self.gmrf.precCondSparse + 1 / self.par.ov2 * PhiTSparse.dot(PhiTSparse.T)  # sequential update of precision matrix

class node:
    def __init__(self, par, gmrf, auv):
        self.gmrf = gmrf
        self.auv = copy.deepcopy(auv)
        self.rewardToNode = 0
        self.accReward = 0
        self.actionToNode = []
        self.depth = 0
        self.parent = []
        self.children = []
        self.visits = 1
        self.D = []
        self.GP = GP(par)
