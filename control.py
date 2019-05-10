import numpy as np
import math
import matplotlib.pyplot as plt
import parameters as par
import methods

class piControl:
    def __init__(self,R,g,lambd,H,K,dt,nUpdated):
        self.R = R
        self.g = g
        self.lambd = lambd
        self.varNoise = self.lambd*np.linalg.inv(R)
        self.H = H
        self.K = K
        self.dt = dt
        self.nUpdated = nUpdated
        self.u0 = np.zeros((self.H,1))
        self.u = self.u0

    def getAction(self, gmrf, x, y):
        self.u0 = self.u
        M = np.dot(np.linalg.inv(self.R), np.dot(self.g, self.g.T))

        for n in range(self.nUpdated):
            noise = np.zeros((self.H, self.K))
            for k in range(self.K):
                xPathRollOut = np.zeros((self.H, self.K))
                yPathRollOut = np.zeros((self.H, self.K))

                # sample control noise and compute path roll-outs
                for j in range(self.H):
                    noise[j,k] = np.random.normal(0, math.sqrt(self.varNoise[j, j]))
                    xPathRollOut[j, k] = x + par.xVel*math.cos(self.u0[j]+noise[j, k])
                    yPathRollOut[j, k] = y + par.yVel*math.sin(self.u0[j]+noise[j, k])

                S = np.zeros((self.H+1, self.K))
                for i in range(self.H):
                    index = self.H-i-1
                    Phi = methods.mapConDis(gmrf, xPathRollOut[index, k], yPathRollOut[index, k])
                    qhh = np.dot(Phi, np.dot(gmrf.covCond, Phi.T))
                    S[index, k] = S[index+1, k] + qhh + 0.5*np.dot((self.u[index, 0]+np.dot(M[index, index], noise[index, k])).T, np.dot(self.R[index, index],self.u[index, 0]+np.dot(M[index, index], noise[index, k])))

            # Compute probability of path segments
            P = np.zeros((self.H, self.K))
            for k in range(self.K):
                for i in range(self.H):
                    probSum = 0
                    for indexSum in range(self.K):
                        probSum += math.exp(-S[i, indexSum]/self.lambd)
                    P[i, k] = math.exp(-S[i, k]/self.lambd)/probSum

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

            self.u = self.u0 + realDeltaU

        xMeas = x + par.xVel * math.cos(self.u[0])
        yMeas = y + par.yVel * math.sin(self.u[0])
        return (xMeas,yMeas)
