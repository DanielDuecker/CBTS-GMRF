# Parameter file for main.py

import numpy as np
import methods
import parameters as par

class gmrf:
    def __init__(self,xMin,xMax,nX,yMin,yMax,nY,nBeta):
        "GMRF properties"
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

        "Mean augmented bayesian regression"
        # Mean regression matrix
        F = np.ones((self.nP,nBeta))
        self.nBeta = nBeta
        self.Tinv = np.linalg.inv(par.valueT*np.eye(self.nBeta))

        # Precision matrix for z values (without regression variable beta)
        self.Lambda = methods.getPrecisionMatrix(self)

        # Augmented prior covariance matrix
        covPriorUpperLeft = np.linalg.inv(self.Lambda)+np.dot(F,np.dot(self.Tinv,F.T))
        covPriorUpperRight = np.dot(F,self.Tinv)
        covPriorLowerLeft = np.dot(F,self.Tinv).T
        covPriorLowerRight = self.Tinv
        self.covPrior = np.vstack( (np.hstack((covPriorUpperLeft,covPriorUpperRight))  ,  np.hstack((covPriorLowerLeft , covPriorLowerRight)) ))

        # Initialize augmented conditioned mean, covariance and precision matrices
        self.meanCond = np.zeros((self.nP+self.nBeta,1))
        self.covCond = self.covPrior
        self.diagCovCond = self.covCond.diagonal().reshape(self.nP+self.nBeta,1)
        self.precCond = np.linalg.inv(self.covCond)

        "Sequential bayesian regression"
        self.bSeq = np.zeros((self.nP+self.nBeta,1))
    
    def bayesianUpdate(self,zMeas,Phi):
        "Update conditioned precision matrix"
        R = np.dot(Phi,np.dot(self.covPrior,Phi.T)) + par.ov2*np.eye(len(zMeas))                    # covariance matrix of measurements
        temp1 = np.dot(Phi,self.covPrior)
        temp2 = np.dot(np.linalg.inv(R),temp1)
        temp3 = np.dot(Phi.T,temp2)
        self.covCond = self.covPrior-np.dot(self.covPrior,temp3)
        #self.covCond = np.linalg.inv((np.linalg.inv(self.covPrior)+1/ov2*np.dot(Phi.T,Phi)))       # alternative way
        self.diagCovCond = self.covCond.diagonal().reshape(self.nP+self.nBeta,1)

        "Update mean"
        self.meanCond = np.dot(self.covPrior,np.dot(Phi.T,np.dot(np.linalg.inv(R),zMeas)))
        #self.meanCond =  1/ov2*np.dot(self.covCond,np.dot(Phi.T,zMeas))                            # alternative way
    
    def seqBayesianUpdate(self,zMeas,Phi):
        Phi_k = Phi[-1,:].reshape(1,self.nP+self.nBeta)                                             # only last measurement mapping is needed
        self.bSeq = self.bSeq + 1/par.ov2 * Phi_k.T*zMeas[-1]                                       # sequential update canonical mean
        self.precCond = self.precCond + 1/par.ov2*np.dot(Phi_k.T,Phi_k)                             # sequential update of precision matrix
        
        hSeq = np.linalg.solve(self.precCond,Phi_k.T)
        self.diagCovCond = self.diagCovCond-np.dot(hSeq,hSeq.T).diagonal().reshape(self.nP+self.nBeta,1)/(par.ov2+np.dot(Phi_k,hSeq))
        # self.diagCovCond = np.linalg.inv(self.precCond).diagonal().reshape(self.nP+self.nBeta,1)  # works, but needs too much time
        self.meanCond = np.dot(np.linalg.inv(self.precCond),self.bSeq)