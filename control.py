import numpy as np
import math
import matplotlib.pyplot as plt
import parameters as par
import methods

class piControl:
    def __init__(self,R,g,varNoise,H,K,dt,nUpdated):
        self.R = R
        self.g = g
        self.varNoise = varNoise
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
            for k in range(self.K):
                noise = np.zeros((self.H, 1))
                pathRollOut = np.zeros((2, self.H))
                # sample control noise and compute path roll-outs
                for j in range(self.H):
                    noise[j] = np.random.normal(0, math.sqrt(self.varNoise[j, j]))
                    pathRollOut[0, j] = x + par.xVel*math.cos(self.u0[j]+noise[j])
                    pathRollOut[1, j] = y + par.yVel*math.sin(self.u0[j]+noise[j])

                S = np.zeros((1,self.H+1))
                for i in range(self.H):
                    index = self.H-i-1
                    Phi = methods.mapConDis(gmrf, pathRollOut[0, index], pathRollOut[1, index])
                    qhh = np.dot(Phi, np.dot(gmrf.covCond, Phi.T))
                    S[0,index] = S[0,index+1] + qhh + 0.5*np.dot((self.u[index,0]+np.dot(M[index,index],noise[index,0])).T,np.dot(self.R[index,index],self.u[index,0]+np.dot(M[index,index],noise[index,0])))
                print(S)

