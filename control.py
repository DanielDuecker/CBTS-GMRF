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
        self.alpha = alpha0 # angle of direction of movement

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
        self.R = R  # input cost matrix
        self.g = g  # mapping from u to states
        self.lambd = lambd  # influences state costs and noise variance
        self.varNoise = self.lambd*np.linalg.inv(self.R)
        self.H = H  # control horizon steps
        self.K = K  # number of path roll outs
        self.dt = dt    # time discretization
        self.nUpdated = nUpdated    # number of iterations
        self.u = np.zeros((self.H, 1))

        self.xTraj = np.zeros((1, self.K))
        self.yTraj = np.zeros((1, self.K))
        self.alphaTraj = np.zeros((1, self.K))
        self.xPathRollOut = np.zeros((1, self.K))
        self.yPathRollOut = np.zeros((1, self.K))

    def getNewState(self, agent, gmrf):
        self.u[0:-2,0] = self.u[1:-1,0]
        self.u[-1,0] = 0

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

                    self.xPathRollOut[:, k] = xTrVec[:, 0]
                    self.yPathRollOut[:, k] = yTrVec[:, 0]

                # compute path costs
                stateCost = 0
                for i in range(self.H):
                    index = self.H-i-1
                    if not methods.sanityCheck(self.xPathRollOut[index, k]*np.eye(1), self.yPathRollOut[index, k]*np.eye(1), gmrf):
                        stateCost += 10*np.amax(1/gmrf.covCond.diagonal())
                    else:
                        Phi = methods.mapConDis(gmrf, self.xPathRollOut[index, k], self.yPathRollOut[index, k])
                        stateCost += 1/np.dot(Phi,gmrf.covCond.diagonal())
                    uHead = self.u[index:self.H,0] + np.dot(M[index:self.H,index:self.H],noise[index:self.H,k])
                    S[index, k] = S[index+1, k] + stateCost + 0.5*np.dot(uHead.T, np.dot(self.R[index:self.H,index:self.H],uHead))

            # Normalize state costs
            S = S/np.amax(S)

            # Compute cost of path segments
            expS = np.zeros((self.H, self.K))
            for k in range(self.K):
                for i in range(self.H):
                    expS[i,k] = math.exp(-S[i, k]/self.lambd)

            P = np.zeros((self.H, self.K))
            for k in range(self.K):
                for i in range(self.H):
                    P[i, k] = expS[i, k] / sum(expS[i,:])

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

        if agent.x < gmrf.xMin:
            agent.x = gmrf.xMin
        elif agent.x > gmrf.xMax:
            agent.x = gmrf.xMax

        if agent.y < gmrf.yMin:
            agent.y = gmrf.yMin
        elif agent.y > gmrf.yMax:
            agent.y = gmrf.yMax

        return (agent.x, agent.y)