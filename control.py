import numpy as np
import math
import matplotlib.pyplot as plt
import parameters as par
import methods
import time

class agent:
    def __init__(self,x0,y0,alpha0):
        self.x = x0
        self.y = y0
        self.alpha = alpha0

    def stateDynamics(self, x, y, alpha, u):
        alpha += u
        x += par.maxStepsize * math.cos(alpha)
        y += par.maxStepsize * math.sin(alpha)
        return x, y, alpha

    def trajectoryFromControl(self,u):
        xTraj = np.zeros((len(u), 1))
        yTraj = np.zeros((len(u), 1))
        alphaTraj = np.zeros((len(u), 1))
        (xTraj[0], yTraj[0], alphaTraj[0]) = (self.x, self.y, self.alpha)
        for i in range(len(u)-1):
            (xTraj[i+1], yTraj[i+1], alphaTraj[i+1]) = self.stateDynamics(xTraj[i], yTraj[i], alphaTraj[i], u[i])
        return xTraj, yTraj, alphaTraj

class piControl:
    def __init__(self, R, g, lambd, H, K, dt, nUpdated):
        self.R = R
        self.g = g
        self.lambd = lambd
        self.varNoise = self.lambd*np.linalg.inv(self.R)
        self.H = H
        self.K = K
        self.dt = dt
        self.nUpdated = nUpdated
        self.u = np.zeros((self.H, 1))

        self.xTraj = np.zeros((1, self.K))
        self.yTraj = np.zeros((1, self.K))
        self.alphaTraj = np.zeros((1, self.K))
        self.xPathRollOut = np.zeros((1, self.K))
        self.yPathRollOut = np.zeros((1, self.K))

    def getNewState(self, agent, gmrf):
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
                    (xTrVec, yTrVec, alphaNew) = agent.trajectoryFromControl(self.u[:, 0] + noise[:, k])

                    # Todo: remove this
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
                    stateCost += np.dot(Phi,1/np.linalg.inv(gmrf.covCond).diagonal())
                    uHead = self.u[index:self.H,0] + np.dot(M[index:self.H,index:self.H],noise[index:self.H,k])
                    S[index, k] = S[index+1, k] + stateCost + 0.5*np.dot(uHead.T, np.dot(self.R[index:self.H,index:self.H],uHead))

            # Compute probability of path segments
            P = np.zeros((self.H, self.K))
            for k in range(self.K):
                for i in range(self.H):
                    probSum = 1e-1000
                    for indexSum in range(self.K):
                        probSum += math.exp(-S[i, indexSum]/self.lambd)
                    P[i, k] = math.exp(-S[i, k]/self.lambd)/probSum

            #for k in range(self.K):
            #    if not methods.sanityCheck(self.xPathRollOut[(self.H-1):self.H, k],self.yPathRollOut[(self.H-1):self.H, k],gmrf):
            #        for i in range(self.H):
            #            rescaling = sum(P[i, :])-P[i, k]
            #            P[i, :] /= rescaling
            #        P[:, k] = np.zeros(self.H)

            # Check if probabilities of path segments add up to 1
            for i in range(self.H):
                if abs(1-sum(P[i, :]))>0.001:
                    print("Warning! Path probabilities don't add up to 1!")

            # Compute next control action
            deltaU = np.zeros((self.H, self.H))
            weigthedDeltaU = np.zeros((self.H, 1))
            for i in range(self.H):
                deltaU[i:self.H, i] = np.dot(np.dot(M[i:self.H, i:self.H], noise[i:self.H,:]),P[i,:].T)
                sumNum = 0
                sumDen = 0
                for h in range(self.H):
                    sumNum += (self.H - h)*deltaU[:, i][i]
                    sumDen += (self.H - h)
                weigthedDeltaU[i, 0] = sumNum/sumDen

            self.u += weigthedDeltaU

        (self.xTraj, self.yTraj, self.alphaTraj) = agent.trajectoryFromControl(self.u)

        # repelling if border is hit
        #if not methods.sanityCheck(self.xTraj[1], self.yTraj[1], gmrf):
        #    self.u[0] += math.pi

        (agent.x, agent.y, agent.alpha) = (self.xTraj[1], self.yTraj[1], self.alphaTraj[1])

        return (agent.x, agent.y)