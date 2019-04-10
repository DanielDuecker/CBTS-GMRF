# Parameter file for main.py

import numpy as np
import methods
import parameters as par

class gmrf:
    def __init__(self,xMin,xMax,nX,yMin,yMax,nY,nBeta):
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
        
        # Regression matrix
        F = np.ones((self.nP,nBeta))
        self.nBeta = nBeta
        self.Tinv = np.linalg.inv(par.valueT*np.eye(self.nBeta))

        # Precision matrix for z values
        self.Lambda = methods.getPrecisionMatrix(self)

        # Augmented prior covariance matrix
        covPriorUpperLeft = np.linalg.inv(self.Lambda)+np.dot(F,np.dot(self.Tinv,F.T))
        covPriorUpperRight = np.dot(F,self.Tinv)
        covPriorLowerLeft = np.dot(F,self.Tinv).T
        covPriorLowerRight = self.Tinv
        self.covPrior = np.vstack( (np.hstack((covPriorUpperLeft,covPriorUpperRight))  ,  np.hstack((covPriorLowerLeft , covPriorLowerRight)) ))

        # Initialize augmented conditioned mean and covariance
        self.meanCond = np.zeros((self.nP+self.nBeta,1))
        self.covCond = np.zeros((self.nP+self.nBeta,self.nP+self.nBeta))
    
    def bayesianUpdate(self,zMeas,ov2,Phi):
        # Update conditioned precision matrix
        R = np.dot(Phi,np.dot(self.covPrior,Phi.T)) + ov2*np.eye(len(zMeas))    # covariance of measurements
        temp1 = np.dot(Phi,self.covPrior)
        temp2 = np.dot(np.linalg.inv(R),temp1)
        temp3 = np.dot(Phi.T,temp2)
        self.covCond = self.covPrior-np.dot(self.covPrior,temp3)

        # Update mean
        self.meanCond = np.dot(self.covCond,np.dot(Phi.T,np.dot(np.linalg.inv(R),zMeas)))