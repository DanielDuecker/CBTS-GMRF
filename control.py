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
            noise = np.zeros((self.K, self.H))
            for k in range(self.K):
                XpathRollOut = np.zeros((self.K, self.H))
                YpathRollOut = np.zeros((self.K, self.H))

                # sample control noise and compute path roll-outs
                for j in range(self.H):
                    noise[k,j] = np.random.normal(0, math.sqrt(self.varNoise[j, j]))
                    XpathRollOut[k, j] = x + par.xVel*math.cos(self.u0[j]+noise[k,j])
                    YpathRollOut[k, j] = y + par.yVel*math.sin(self.u0[j]+noise[k,j])

                S = np.zeros((self.K,self.H+1))
                for i in range(self.H):
                    index = self.H-i-1
                    Phi = methods.mapConDis(gmrf, XpathRollOut[k, index], YpathRollOut[k, index])
                    qhh = np.dot(Phi, np.dot(gmrf.covCond, Phi.T))
                    S[k, index] = S[k, index+1] + qhh + 0.5*np.dot((self.u[index, 0]+np.dot(M[index, index], noise[k, index])).T, np.dot(self.R[index, index],self.u[index, 0]+np.dot(M[index, index], noise[k,index])))

            # Compute probability of path segments
            P = np.zeros((self.K, self.H))
            for k in range(self.K):
                for i in range(self.H):
                    probSum = 0
                    for indexSum in range (self.K):
                        probSum += math.exp(-S[indexSum, i]/self.lambd)
                    P[k, i] = math.exp(-S[k, i]/self.lambd)/probSum

            # Compute next control action
            deltaU = np.zeros((self.H-1, 1))
            for i in range(self.H-1):
                for k in range(self.K):
                    print(P[k, i].shape)
                    print(np.dot(M[i:self.H, i:self.H], noise[k, i:self.H]).shape)
                    print(M[i:self.H, i:self.H].shape)
                    print(noise[k, i:self.H].shape)
                    deltaU[i, 0] += P[k, i]*np.dot(M[i:self.H, i:self.H], noise[k, i:self.H])