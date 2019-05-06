import numpy as np
import math
import matplotlib.pyplot as plt
import parameters as par

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

    def getAction(self, x, y):
        self.u0 = self.u
        M = np.dot(np.linalg.inv(self.R), np.dot(self.g, self.g.T))

        for n in range(self.nUpdated):
            for k in range(self.K):
                fig2 = plt.figure()
                noise = np.zeros((self.H, 1))
                pathRollOut = np.zeros((2, self.H))
                # sample control noise and compute path roll-outs
                for j in range(self.H):
                    noise[j] = np.random.normal(0, math.sqrt(self.varNoise[j, j]))
                    pathRollOut[0, j] = x + par.xVel*math.cos(self.u0[j]+noise[j])
                    pathRollOut[1, j] = y + par.yVel*math.sin(self.u0[j]+noise[j])

                for i in range(H):
                    test=1
                plt.plot(pathRollOut[0, :], pathRollOut[1, :])
                plt.show(block=True)
                while(1):
                    t=1
