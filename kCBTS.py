import math
import numpy as np
import methods
import copy

class node:
    def __init__(self,gmrf,auv,r):
        self.gmrf = copy.copy(gmrf)
        self.auv = copy.copy(auv)
        self.totalR = copy.copy(r)
        self.depth = 0
        self.parent = []
        self.children = []
        self.visits = 1
        self.D = []

class kCBTS:
    def __init__(self, nIterations, nTrajPoints, maxParamExploration, maxDepth, branchingFactor, kappa):
        self.nIterations = nIterations
        self.nTrajPoints = nTrajPoints
        self.maxParamExploration = maxParamExploration
        self.maxDepth = maxDepth
        self.branchingFactor = branchingFactor # maximum number of generated actions per node
        self.kappa = kappa

    def getNewState(self, auv, gmrf):
        v0 = node(gmrf,auv,0) # create node with belief b and total reward 0
        for i in range(self.nIterations):
            print("____")
            print("kCBTS-Iteration",i,"of",self.nIterations)
            vl = self.treePolicy(v0) # get next node
            r = self.exploreNode(vl)
            self.backUp(v0,vl,r)
        bestTraj, auv.alpha = self.getBestTheta(v0)
        return bestTraj

    def treePolicy(self,v):
        print(" call tree policy:")
        while v.depth < self.maxDepth:
            if len(v.D) < self.branchingFactor:
                print("     generate new node at depth ",v.depth," and action number ",len(v.D))
                theta = self.getNextTheta(v.D)
                traj, alphaEnd = self.generateTrajectory(v, theta)
                print("generated trajectory: ",traj)
                print("with theta = ",theta)
                print("data set is now: ",v.D)

                r,o = self.evaluateTrajectory(v,traj)
                v.D.append((theta,r))

                vNew = copy.copy(v)
                vNew.totalR += r
                vNew.parent = v
                v.children.append(vNew)
                print("Current children of node:",v.children)
                vNew.depth = v.depth + 1

                # simulate GP update
                for i in range(len(o)):
                    vNew.auv.x = traj[0,i+1]
                    vNew.auv.y = traj[1,i+1]
                    Phi = methods.mapConDis(vNew.gmrf,vNew.auv.x,vNew.auv.y)
                    vNew.gmrf.seqBayesianUpdate(o[i],Phi)
                vNew.auv.alpha = alphaEnd

                return vNew
            else:
                print(v)
                v = self.bestChild(v)
                print(v)
                print("No more actions, go to best child and check available actions")

    def getNextTheta(self,Dv):
        # Todo: Use Uppder Confidence Bound (Ramos 2019)
        maxR = -math.inf
        bestTheta = np.random.rand(5)*self.maxParamExploration
        for theta,r in Dv:
            if r > maxR:
                bestTheta = theta
                maxR = r
        return bestTheta

    def exploreNode(self,vl):
        r = 0
        v = copy.copy(vl)
        while v.depth < self.maxDepth:
            nextTheta = np.random.rand(5)*self.maxParamExploration
            nextTraj, alphaEnd = self.generateTrajectory(v,nextTheta)

            dr,do = self.evaluateTrajectory(v,nextTraj)
            r += dr

            v.auv.x = nextTraj[0,-1]
            v.auv.y = nextTraj[1,-1]
            v.auv.alpha = alphaEnd
            v.depth += 1
        return r

    def getBestTheta(self,v0):
        maxR = -math.inf
        bestTheta = np.random.rand(5)*self.maxParamExploration
        for theta,r in v0.D:
            if r > maxR:
                bestTheta = theta
                maxR = r
        bestTraj, alphaEnd = self.generateTrajectory(v0,bestTheta)
        return bestTraj, alphaEnd


    def generateTrajectory(self,v,theta):
        # theta = [ax ay bx by cx]
        # beta =    [dx cx bx ax]
        #           [dy cy by ay]
        # dx = posX, dy = posY, cy/cx = tan(alpha)
        ax = theta[0]
        ay = theta[1]
        bx = theta[2]
        by = theta[3]
        cx = theta[4]
        cy = cx * math.tan(v.auv.alpha)
        dx = v.auv.x
        dy = v.auv.y

        beta = np.array([[dx,cx,bx,ax],[dy,cy,by,ay]])

        alphaEnd = math.tanh((3*ay+2*by+cy)/(3*ax+2*bx+cx))

        tau = np.zeros((2,self.nTrajPoints))
        for i in range(self.nTrajPoints):
            u = i/self.nTrajPoints
            tau[:,i] = np.dot(beta,np.array([[1],[u],[u**2],[u**3]]))[:,0]
        return tau, alphaEnd

    def evaluateTrajectory(self,v,tau):
        # TODO: maybe use r = sum(grad(mue) + parameter*sigma) from Seq.BO paper (Ramos)
        r = 0
        o = []
        for i in range(self.nTrajPoints-1):
            Phi = methods.mapConDis(v.gmrf, tau[0,i+1], tau[1,i+1])
            r += np.dot(Phi,v.gmrf.covCond.diagonal())
            o.append(np.dot(Phi,v.gmrf.meanCond))
        return r,o

    def backUp(self,v0,v,r):
        nodePointer = v
        while nodePointer != v0:
            nodePointer.visits += 1
            nodePointer.totalR += r
            nodePointer = nodePointer.parent

    def bestChild(self,v):
        g = []
        for child in v.children:
            g.append(child.totalR/child.visits + self.kappa*math.sqrt(2*math.log(v.visits)/child.visits))
        print("Trying to find best children:")
        return v.children[np.argmax(g)]
