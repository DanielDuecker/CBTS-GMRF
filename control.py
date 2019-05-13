import numpy as np
import math
import matplotlib.pyplot as plt
import parameters as par
import methods
import time

class piControl:
    def __init__(self,R,g,lambd,H,K,dt,nUpdated):
        self.R = R
        self.g = g
        self.lambd = lambd
        self.varNoise = self.lambd*np.linalg.inv(self.R)
        self.H = H
        self.K = K
        self.dt = dt
        self.nUpdated = nUpdated
        self.u = math.pi / 4*np.ones((self.H, 1))

        self.xTraj = np.zeros((1, self.K))
        self.yTraj = np.zeros((1, self.K))
        self.xPathRollOut = np.zeros((1, self.K))
        self.yPathRollOut = np.zeros((1, self.K))

    def stateDynamics(self,x,y,u):
        xNext = x + par.xVel * math.cos(u)
        yNext = y + par.yVel * math.sin(u)
        return xNext, yNext

    def trajectoryFromControl(self,x,y,u):
        xTraj = np.zeros((len(u), 1))
        yTraj = np.zeros((len(u), 1))
        (xTraj[0],yTraj[0]) = (x, y)
        for i in range(len(u)-1):
            (xTraj[i+1],yTraj[i+1]) = self.stateDynamics(xTraj[i],yTraj[i],u[i])
        return xTraj, yTraj

    def getNewState(self, gmrf, x, y):
        M = np.dot(np.linalg.inv(self.R), np.dot(self.g, self.g.T))/(np.dot(self.g.T, np.dot(np.linalg.inv(self.R), self.g)))
        for n in range(self.nUpdated):
            noise = np.zeros((self.H, self.K))
            self.xPathRollOut = np.zeros((self.H, self.K))
            self.yPathRollOut = np.zeros((self.H, self.K))

            S = np.zeros((self.H + 1, self.K))
            for k in range(self.K):
                # sample control noise and compute path roll-outs
                for j in range(self.H):
                    noise[j, k] = np.random.normal(0, math.sqrt(self.varNoise[j, j]))
                    (xTrVec, yTrVec) = self.trajectoryFromControl(x, y, self.u[:, 0] + noise[:, k])

                    # repeat if states are out of bound
                    #while not methods.sanityCheck(xTrVec[:, 0], yTrVec[:, 0], gmrf):
                    #    noise[j, k] = np.random.normal(math.pi/4, 2*math.sqrt(self.varNoise[j, j]))
                    #    (xTrVec, yTrVec) = self.trajectoryFromControl(x, y, self.u[:, 0] + noise[:, k])

                    self.xPathRollOut[:, k] = xTrVec[:, 0]
                    self.yPathRollOut[:, k] = yTrVec[:, 0]

                # compute path costs
                stateCost = 0
                for i in range(self.H):
                    index = self.H-i-1
                    Phi = methods.mapConDis(gmrf, self.xPathRollOut[index, k], self.yPathRollOut[index, k])
                    stateCost += np.dot(Phi, np.dot(np.linalg.inv(gmrf.covCond), Phi.T))
                    uHead = self.u[index:self.H,0] + np.dot(M[index:self.H,index:self.H],noise[index:self.H,k])
                    S[index, k] = S[index+1, k] + stateCost + 0.5*np.dot(uHead.T, np.dot(self.R[index:self.H,index:self.H],uHead))

            # Compute probability of path segments
            P = np.zeros((self.H, self.K))
            for k in range(self.K):
                for i in range(self.H):
                    probSum = 1e-10000
                    for indexSum in range(self.K):
                        probSum += math.exp(-S[i, indexSum]/self.lambd)
                    P[i, k] = math.exp(-S[i, k]/self.lambd)/probSum

            # Check if probabilities of path segments add up to 1
            for i in range(self.H):
                if abs(1-sum(P[i, :]))>0.001:
                    input("Warning! Path probabilities don't add up to 1!")

            # Compute next control action
            deltaU = np.zeros((self.H, self.H))
            for i in range(self.H):
                deltaU[i:self.H,i] += np.dot(np.dot(M[i:self.H, i:self.H], noise[i:self.H,:]),P[i,:].T)

            realDeltaU = np.zeros((self.H, 1))
            for i in range(self.H):
                sumNum = 0
                sumDen = 0
                for h in range(self.H):
                    sumNum += (self.H - h)*deltaU[:, i][i]
                    sumDen += (self.H - h)
                realDeltaU[i, 0] = sumNum/sumDen

            self.u += realDeltaU

        (self.xTraj, self.yTraj) = self.trajectoryFromControl(x, y, self.u)

        if not methods.sanityCheck(self.xTraj[1], self.yTraj[1], gmrf):
            self.u[0] += math.pi

        (self.xTraj, self.yTraj) = self.trajectoryFromControl(x, y, self.u)
        (xNext, yNext) = (self.xTraj[1], self.yTraj[1])

        return (xNext, yNext)
