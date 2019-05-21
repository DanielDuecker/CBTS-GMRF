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

    def generateTrajectory(self,theta,pos):
        # create anchor points
        lMax = 10
        eps = 1
        delta = 1
        Traj = np.zeros((2,self.nAnchorPoints))

        anchorPoints = np.zeros((2,self.nAnchorPoints))
        anchorPoints[:,0] = pos
        y = np.zeros((1,self.nAnchorPoints))
        y[0,0] = 0

        for i in range(self.nAnchorPoints-1):
            alpha = 2*pi*np.random.normal(self.trajectoryNoise)
            anchorPoints[0,i+1] = anchorPoints[0,i] + par.maxStepsize/self.nAnchorPoints * math.cos(alpha))
            anchorPoints[1,i+1] = anchorPoints[1,i] + par.maxStepsize/self.nAnchorPoints * math.sin(alpha))
            y[0,i+1] = par.dt/self.nAnchorPoints
        # calculate weight vector
        GX = np.zeros((self.nAnchorPoints,self.nAnchorPoints))
        GY = np.zeros((self.nAnchorPoints,self.nAnchorPoints))
        mPi = np.zeros((self.nAnchorPoints,1))

        for i in range(self.nAnchorPoints):
            for j in range(self.nAnchorPoints):
                GX[i,j] = self.RBFkernel(anchorPoints[:,i],anchorPoints[:,j])
                GY[i,j] = self.RBFkernel(y[0,i],y[0,j])
            mPi[i,0] = sum(np.random.choice(GX),lMax)/lMax

        w = np.zeros((self.nAnchorPoints,1))
        Lambd = np.dot(np.linalg.inv(GX + eps*np.eye(self.nAnchorPoints)),mPi)
        LambdGY = np.dot(Lambd.T,GY)
        for y in range(self.nAnchorPoints):
            w[y] = np.dot(LambdGY,np.dot(np.linalg.inv(np.dot(LambdGY,LambdGY.T)+eps),np.dot(Lambd,GY[:,y])))
            Traj[y] = sum(np.dot(anchorPoints,w))

    def RBFkernel(self,vec1,vec2):
        return math.exp(-(np.linalg.norm(vec1-vec2)**2)/(2*self.sigmaKernel**2))



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
