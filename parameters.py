# Parameter file for main.py

import numpy as np
import methods

class gmrf:
    def __init__(self,xMin,xMax,nX,yMin,yMax,nY,nBeta,valT):
        self.xMin = xMin
        self.xMax = xMax
        self.nX = nX                                        # Total number of vertices in x
        self.x = np.linspace(self.xMin,self.xMax,self.nX)   # Vector of x grid values

        self.yMin = yMin
        self.yMax = yMax
        self.nY = nY                                        # Total number of vertices in y 
        self.y = np.linspace(self.yMin,self.yMax,self.nY)   # Vector of y grid values

        self.nP = nX*nY                                     # Total number of vertices

        # Distance between two vertices in x and y
        self.dx = (self.xMax-self.xMin)/(self.nX-1)
        self.dy = (self.yMax-self.yMin)/(self.nY-1)

        # Extended field model:
        # Field  values z = mu(x,y,beta)+eta
        # with  mu(x,y,beta) = f(x,y).T*beta
        #       eta ~ N(0,inv(precCondEta))
        #       beta ~ N(0,inv(T))  
        
        # Prior mean and precision matrix
        self.mu = np.ones((self.nP,1))       # Prior field means
        self.beta = np.zeros((nBeta,1))      # Prior beta coefficients
        self.zBar = np.vstack((self.mu,self.beta))      # combined vector

        self.Q = methods.getPrecisionMatrix(self)
        self.Tinv = valT*np.eye(nBeta)

        # Augmented mean and precision matrix
        self.precCondEta = np.eye(nY*nX,nY*nX)

        F = np.ones((self.nP,nBeta))

        self.covPrior = np.vstack(( np.hstack(( self.Q+np.multiply( np.multiply(F,self.Tinv),F.T ) , np.multiply(F,self.Tinv) ))  ,  np.hstack(( (np.multiply(F,self.Tinv) ).T , self.Tinv)) ))
        self.covCond = np.zeros((self.nP+nBeta,self.nP+nBeta))

    def bayesianUpdate(zMeas,oz2,phi):

        # Update conditioned precision matrix
        R = np.multiply(phi,np.multiply(covPrior,phi.T)) + oz2*np.eye(len(zMeas))
        temp1 = np.multiply(self.covPrior,phi)
        temp2 = np.multiply(np.linalg.inv(R),temp1)
        temp3 = np.multiply(phi.T,temp2)
        self.covCond = self.covPrior-np.multiply(self.covPrior,temp3)

        # Update mean
        self.meanCond = np.multiply(self.covCond,np.multiply(phi.T,np.multiply(np.linalg.inv(R),zMeas)))