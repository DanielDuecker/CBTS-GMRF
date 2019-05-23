import math
import numpy as np
import parameters as par

class node:
    def __index__(self,b,r):
        self.belief = b
        self.totalR = r
        self.depth = 0
        self.parent  = []
        self.children = []
        self.visits = 0

class kcBTS:
    def __init__(self,nIterations, nAnchorPoints,trajectoryNoise, maxDepth, aMax, kappa, sigmaKernel):
        self.nIterations = nIterations
        self.nAnchorPoints = nAnchorPoints
        self.trajectoryNoise = trajectoryNoise
        self.maxDepth = maxDepth
        self.aMax = aMax # maximum number of generated actions per node
        self.kappa = kappa
        self.sigmaKernel = sigmaKernel

    def kCBTS(self,pos,b)
        v0 = node(b,0) # create node with belief b and total reward 0
        for i in range(self.nIterations):
            vl = self.treePolicy(v0,pos) # get next node
            r = self.getActions(vl,self.maxDepth)
            self.backUp(v0,vl,r)
        return self.argmax(v0)

    def treePolicy(self,v,pos):
        Dv = []
        while v.depth < maxDepth:
            if len(Dv) < self.aMax:
                bestTheta = self.getBestTheta
                tau = self.generateTrajectory(bestTheta,pos)
                r,o = self.evalTrajectory(tau)
                Dv.append(bestTheta,r)
                return node(b,r)
            else:
                return self.bestChild(v)

    def getTrajectoryPoint(self,theta,pos,alpha,u):
        # theta = [ax ay bx by cx]
        # beta =    [dx cx bx ax]
        #           [dy cy by ay]
        # dx = posX, dy = posY, cy/cx = tan(alpha)
        beta = np.array([[pos[0],theta[4],theta[2],theta[0]],[pos[1],theta[4]*math.tan(alpha),theta[3],theta[1]]])
        return np.dot(beta,np.array([[1],[u],[u**2],[u**3]]))


    def backUp(self,v0,v,r):
        while v != v0:
            v.visits += 1
            v.totalR += r
            v = v.parent

    def bestChild(self,v):
        g=[]
        for child in v.children:
            g.append(child.totalR/child.visits + self.kappa*math.sqrt(2*math.log(v.visits)/child.visits))
        return v.children[np.argmax(g)]
