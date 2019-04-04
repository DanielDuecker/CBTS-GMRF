# Parameter file for main.py

import numpy as np

class gmrf:
    def __init__(self,xMin,xMax,nX,yMin,yMax,nY):
        self.xMin = xMin
        self.xMax = xMax
        self.nX = nX                                        # Total number of vertices in x
        self.x = np.linspace(self.xMin,self.xMax,self.nX)   # Vector of x grid values

        self.yMin = yMin
        self.yMax = yMax
        self.nY = nY                                        # Total number of vertices in y 
        self.y = np.linspace(self.yMin,self.yMax,self.nY)   # Vector of y grid values

        self.muCond = np.array([np.zeros((nY,nX)).flatten()]).T
        self.precCond = np.eye(nY*nX,nY*nX)
        self.nP = nX*nY                                     # Total number of vertices

        # Distance between two vertices in x and y
        self.dx = (self.xMax-self.xMin)/(self.nX-1)
        self.dy = (self.yMax-self.yMin)/(self.nY-1)