import numpy as np
import copy
import parameters as par

class node:
    def __init__(self,gmrf,auv,r):
        self.gmrf = copy.deepcopy(gmrf)
        self.auv = copy.deepcopy(auv)
        self.totalR = copy.deepcopy(r)
        self.depth = 0
        self.parent = []
        self.children = []
        self.visits = 1
        self.D = []

class mapActionReward:
    def __init__(self,thetaMin,thetaMax,nMapping,trajOrder):
        self.thetaMin = thetaMin
        self.thetaMax = thetaMax
        self.nMapping = nMapping
        self.nGridPoints = nMapping**trajOrder
        self.trajOrder = trajOrder
        self.meanCond = np.zeros((self.nGridPoints,1))
        self.cov = np.eye(self.nGridPoints)
        self.prec = np.linalg.inv(self.cov)
        self.precCond = self.prec
        self.covCond = self.cov
        self.bSeq = np.zeros((self.nGridPoints,1))
        self.thetaRange = np.linspace(thetaMin, thetaMax, self.nMapping+1)
        self.gridSize = self.thetaRange[1]-self.thetaRange[0]

    def resetMapping(self):
        self.meanCond = np.zeros((self.nGridPoints,1))
        self.cov = np.eye(self.nGridPoints)
        self.prec = np.linalg.inv(self.cov)
        self.precCond = self.prec
        self.covCond = self.cov
        self.bSeq = np.zeros((self.nGridPoints,1))

    def getIntervalIndex(self,thetaValue):
        for i in range(self.nMapping):
            if self.thetaRange[i] <= thetaValue < self.thetaRange[i+1]:
                return i
        if self.thetaMax <= thetaValue:
            return self.nMapping-1
        elif thetaValue <= self.thetaMin:
            return 0
        return "not in interval"

    def mapConDisAction(self,theta):
        # Phi = [[theta_n_0],[theta_n_1],[theta_n_2],...]
        # with theta_n_i = [[theta_n-1_0],[theta_n-1_1],[theta_n-1_2],...] and so on
        Phi = np.zeros((1,self.nGridPoints))
        index = np.zeros((self.trajOrder,1))
        for i in range(self.trajOrder):
            index[i] = self.getIntervalIndex(theta[0,i]) * self.nMapping**i
        Phi[0,int(sum(index))] = 1
        return Phi

    def updateMapActionReward(self,theta,z):
        Phi = self.mapConDisAction(theta)
        self.bSeq = self.bSeq + 1 / par.ovMap2 * Phi.T * z  # sequential update canonical mean
        self.precCond = self.precCond + 1 / par.ovMap2 * np.dot(Phi.T, Phi)  # sequential update of precision matrix
        self.meanCond = np.dot(np.linalg.inv(self.precCond), self.bSeq)
        self.covCond = np.linalg.inv(self.precCond)
        self.diagCovCond = self.covCond.diagonal()

    def convertIndextoTheta(self,index):
        theta = np.zeros((1, self.trajOrder))
        for i in range(self.trajOrder):
            position = int(index / (self.nMapping ** (self.trajOrder - i - 1)))
            positionIndex = self.trajOrder - i - 1
            theta[0, positionIndex] = self.thetaRange[position]
            index = index % (self.nMapping ** (self.trajOrder - i - 1))
        return theta

