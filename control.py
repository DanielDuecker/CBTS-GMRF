import numpy as np
import math

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

    def getAction(self):
        self.u0 = self.u
        M = np.dot(np.linalg.inv(self.R),np.dot(self.g,self.g.T))

        for i in range(self.nUpdated):
            for k in range(self.K):
                noise = np.zeros((self.H,1))
                for j in range(self.H):
                    noise[j] = np.random.normal(0, math.sqrt(self.varNoise[j,j]))
