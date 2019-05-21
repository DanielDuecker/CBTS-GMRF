import math

class node:
    def __index__(self,b,r):
        self.belief = b
        self.totalR = r
        self.depth = 0
        self.parent  = []
        self.children = []
        self.visits = 0

class kcBTS:
    def __init__(self,nIterations, nAnchorPoints, maxDepth, aMax, kappa):
        self.nIterations = nIterations
        self.nAnchorPoints = nAnchorPoints
        self.maxDepth = maxDepth
        self.aMax = aMax # maximum number of generated actions per node
        self.kappa = kappa

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

    def generateTrajectory(theta,pos):
        # create anchor points

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
