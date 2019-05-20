import numpy as np
import math
import parameters as par
import methods

def stateDynamics(self, x, y, alpha, u):
    alpha += u
    x += par.maxStepsize * math.cos(alpha)
    y += par.maxStepsize * math.sin(alpha)
    return x, y, alpha

class piControl:
    def __init__(self, H, nUpdated):
        self.H = H
        self.nUpdated = nUpdated
        self.a = 1

    def calcEntropy(self, covMatrix):
        return 0.5*math.log(np.trace(covMatrix))*self.a**(1/(self.a-1))

    def utility(self,gmrf,newX,newY):
        oldCovMatrix = gmrf.covCond

        Phi = methods.mapConDis(gmrf, newX, newY)
        precCond = gmrf.precCond + 1 / par.ov2 * np.dot(Phi.T, Phi)
        newCovMatrix = np.linalg.inv(precCond).diagonal().reshape(gmrf.nP + gmrf.nBeta, 1)

        return self.calcEntropy(oldCovMatrix) - self.calcEntropy(newCovMatrix)

    def getBestState(self):


    def replanPath(self,x,y, gmrf):
        meanCopy = gmrf.meanCond
        covCopy = gmrf.covCond
        trajX = x*np.eye(1)
        trajY = y*np.eye(1)

        for i in range(self.nUpdated):
            nextX,nextY = self.getBestState()
            trajX.append(nextX)
            trajY.append(nextY)

        return x,y
