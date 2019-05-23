import math
import numpy as np
import classes
import methods

class node:
    def __index__(self,b,r):
        self.belief = b
        self.totalR = r
        self.depth = 0
        self.parent  = []
        self.children = []
        self.visits = 0
        self.actionToNode = []

class kCBTS:
    def __init__(self,gmrf1, nIterations, nAnchorPoints,trajectoryNoise, maxParamExploration, maxDepth, aMax, kappa):
        self.gmrf = classes.gmrf(gmrf1.xMin,gmrf1.xMax,gmrf1.nX,gmrf1.yMin,gmrf1.nY,gmrf1.nBeta)
        self.gmrf.meanCond = gmrf1.meanCond
        self.gmrf.covCond = gmrf1.covCond
        self.gmrf.bSeq = gmrf1.bSeq
        self.gmrf.precCond = gmrf1.precCond

        self.nIterations = nIterations
        self.nAnchorPoints = nAnchorPoints
        self.trajectoryNoise = trajectoryNoise
        self.maxParamExploration = maxParamExploration
        self.maxDepth = maxDepth
        self.aMax = aMax # maximum number of generated actions per node
        self.kappa = kappa

    def getNewState(self,pos,alpha,b,cov)
        # Copy belief and covariance
        self.gmrf.meanCond = b
        self.gmrf.covCond = cov

        v0 = node(b,0) # create node with belief b and total reward 0
        for i in range(self.nIterations):
            vl = self.treePolicy(v0,pos,alpha) # get next node
            r = self.exploreNode(vl,pos,alpha)
            self.backUp(v0,vl,r)
        return self.argmax(v0,pos,alpha)

    def treePolicy(self,v,pos,alpha):
        Dv = []
        while v.depth < maxDepth:
            if len(Dv) < self.aMax:
                bestTheta = self.getBestTheta
                tau = self.generateTrajectory(bestTheta,pos,alpha)
                r,o = self.evalTrajectory(self.gmrf,tau)

                # simulate GP update
                Phi = methods.mapConDis(self.gmrf,tau[0,1],tau[1,1])
                self.gmrf.seqBayesianUpdate(o,Phi)
                self.gmrf.covCond = np.linalg.inv(self.gmrf.precCond)

                Dv.append(bestTheta,r)
                vNew = node(self.gmrf.meanCond,r)
                vNew.actionToNode = bestTheta
                v.children = vNew
                return vNew
            else:
                return self.bestChild(v)

    def exploreNode(self,vl,pos,alpha):
        r = 0
        while vl.depth < self.maxDepth:
            nextTheta = np.random.rand(1,5)*self.maxParamExploration
            nextTau = self.generateTrajectory(nextTheta,pos,alpha)
            r += self.evaluateTrajectory(nextTau)
            pos = nextTau[:,-1]
            alpha = math.atan((3*nextTheta[1]+2*nextTheta[3]+nextTheta[4]*math.tan(alpha))/(3*nextTheta[0]+2*nextTheta[2]+nextTheta[4]))
        return r

    def argmax(self,v0,pos,alpha):
        R = 0
        for child in v0.children:
            if child.totalR > R:
                bestAction = child.actionToNode
        bestTraj = self.generateTrajectory(bestAction,pos,alpha)
        alpha = math.atan((3 * bestAction[1] + 2 * bestAction[3] + bestAction[4] * math.tan(alpha)) / (
                    3 * bestAction[0] + 2 * bestAction[2] + bestAction[4]))
        return bestTraj[:,1],alpha


    def generateTrajectory(self,theta,pos,alpha):
        # theta = [ax ay bx by cx]
        # beta =    [dx cx bx ax]
        #           [dy cy by ay]
        # dx = posX, dy = posY, cy/cx = tan(alpha)
        beta = np.array([[pos[0],theta[4],theta[2],theta[0]],[pos[1],theta[4]*math.tan(alpha),theta[3],theta[1]]])

        tau = np.zeros((2,self.nAnchorPoints))
        for i in range(self.nAnchorPoints):
            u = i/self.nAnchorPoints
            tau[:,i] = np.dot(beta,np.array([[1],[u],[u**2],[u**3]]))
        return tau

    def evaluateTrajectory(self,tau):
        # TODO: maybe use r = sum(grad(mue) + parameter*sigma) from Seq.BO paper (Ramos)
        r = 0
        for i in range(self.nAnchorPoints):
            Phi = methods.mapConDis(self.gmrf, tau[0,i], tau[1,i])
            r += np.dot(Phi,self.gmrf.covCond.diagonal())
        return r

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
