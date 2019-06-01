import math
import numpy as np
import methods
import copy
import parameters as par
import matplotlib.pyplot as plt
import classes

class node:
    def __init__(self,gmrf,auv,r):
        self.gmrf = copy.deepcopy(gmrf)
        self.auv = copy.deepcopy(auv)
        self.totalR = copy.deepcopy(r)
        self.depth = 0
        self.parent = []
        self.children = []
        self.visits = 1
        self.D = []

class mapActionReward:
    def __init__(self,thetaMin,thetaMax,nMapping,trajOrder):
        self.nMapping = nMapping
        self.nGridPoints = nMapping*trajOrder
        self.trajOrder = trajOrder
        self.meanCond = np.zeros((self.nGridPoints,1))
        self.cov = 0.2*np.ones((self.nGridPoints,self.nGridPoints))+0.8*np.eye(self.nGridPoints)
        self.prec = np.linalg.inv(self.cov)
        self.precCond = self.prec
        self.covCond = self.cov
        self.bSeq = np.zeros((self.nGridPoints,1))
        self.thetaRange = np.linspace(thetaMin, thetaMax, self.nMapping)
        self.gridSize = self.thetaRange[1]-self.thetaRange[0]

    def getIntervalIndex(self,thetaValue):
        for i in range(self.nMapping-1):
            if self.thetaRange[i] <= thetaValue < self.thetaRange[i+1]:
                print(self.thetaRange[i])
                print(self.thetaRange[i+1])
                print(thetaValue)
                #todo check index (still not correct)
                return i
        return "not in interval"

    def mapConDisAction(self,theta):
        # Phi = [[theta_n_0],[theta_n_1],[theta_n_2],...]
        # with theta_n_i = [[theta_n-1_0],[theta_n-1_1],[theta_n-1_2],...] and so on
        Phi = np.zeros((1,self.nGridPoints))
        index = np.zeros((self.trajOrder,1))
        for i in range(self.trajOrder):
            index[i] = self.getIntervalIndex(theta[i]) * self.nMapping**i
        Phi[int(sum(index))] = 1
        print("***Called mapConDisAction***")
        print("theta:",theta)
        print("index:",index)
        return Phi

    def updateMapActionReward(self,theta,z):
        Phi = self.mapConDisAction(theta)
        self.bSeq = self.bSeq + 1 / par.ovMap2 * Phi * z  # sequential update canonical mean
        self.precCond = self.precCond + 1 / par.ovMap2 * np.dot(Phi.T, Phi)  # sequential update of precision matrix
        self.meanCond = np.dot(np.linalg.inv(self.precCond), self.bSeq)
        self.covCond = np.linalg.inv(self.precCond)

    def getCovarianceFromAction(self,theta):
        Phi = self.mapConDisAction(theta)
        return np.dot(Phi,np.dot(self.covCond,Phi.T))

    def getBestTheta(self):
        index = np.argmax(self.meanCond)
        theta = np.zeros((1,self.trajOrder))
        for i in range(self.trajOrder):
            theta[i] = self.thetaRange[index % (self.nMapping**i)]
        print("Best Theta called. Best theta at index",index)
        print("That's theta = ",theta)
        return theta


class kCBTS:
    def __init__(self, nIterations, nTrajPoints, maxParamExploration, trajOrder, maxDepth, branchingFactor, kappa):
        self.nIterations = nIterations
        self.nTrajPoints = nTrajPoints
        self.maxParamExploration = maxParamExploration
        self.trajOrder = trajOrder
        self.maxDepth = maxDepth
        self.branchingFactor = branchingFactor # maximum number of generated actions per node
        self.kappa = kappa

    def getNewTraj(self, auv, gmrf):
        v0 = node(gmrf,auv,0) # create node with belief b and total reward 0
        #figTest = plt.figure()
        #plt.show()
        for i in range(self.nIterations):
            print("kCBTS-Iteration",i,"of",self.nIterations)
            vl = self.treePolicy(v0) # get next node
            #vl = self.treePolicy(figTest,v0) # get next node
            if vl == None:
                continue # max depth and branching reached
            r = self.exploreNode(vl)
            print("exploring node ",vl," at position",vl.auv.x,vl.auv.y, " yields reward of",r)
            self.backUp(v0,vl,vl.totalR+r)
            print("Level 1 nodes after backing up:")
            for Eachnode in v0.children:
                print("Node:",Eachnode,"/Reward: ",Eachnode.totalR,"/Counter: ",Eachnode.visits)
            print("Best trajectory is now returned")
            print("_______________________________")

        bestTraj, auv.alpha = self.getBestTheta(v0)
        return bestTraj

    def treePolicy(self,v):
    #def treePolicy(self, figTest, v):
        print(" call tree policy:")
        while v.depth < self.maxDepth:
            if len(v.D) < self.branchingFactor:
                print("     generate new node at depth ",v.depth)
                theta = self.getNextTheta(v.D)
                traj, alphaEnd = self.generateTrajectory(v, theta)
                #plt.plot(traj[0, :], traj[1, :])
                #figTest.canvas.draw()

                r,o = self.evaluateTrajectory(v,traj)
                v.D.append((theta,r))

                print("     generated trajectory: ",traj)
                print("     with theta = ",theta)
                print("     data set is now: ",v.D)
                print("     reward is: ",r)

                # Update GP mapping from theta to r:
                # todo

                # Create new node:
                vNew = node(v.gmrf,v.auv,v.totalR)
                vNew.totalR += r
                vNew.parent = v
                vNew.depth = v.depth + 1

                v.children.append(vNew)

                # simulate GP update of belief
                for i in range(len(o)):
                    vNew.auv.x = traj[0,i+1]
                    vNew.auv.y = traj[1,i+1]
                    Phi = methods.mapConDis(vNew.gmrf,vNew.auv.x,vNew.auv.y)
                    vNew.gmrf.seqBayesianUpdate(o[i],Phi)
                vNew.auv.alpha = alphaEnd
                return vNew
            else:
                print("No more actions. Switching from node",v)
                v = self.bestChild(v)
                print("to node",v)
                #print("to node ",v)

    def getNextTheta(self,Dv):
        # Todo: Use Uppder Confidence Bound (Ramos 2019)
        #maxR = -math.inf
        bestTheta = np.random.normal(1,self.maxParamExploration,self.trajOrder)
        #for theta,r in Dv:
        #    if r > maxR:
        #        bestTheta = theta
        #        maxR = r
        return bestTheta

    def exploreNode(self,vl):
        r = 0
        v = copy.deepcopy(vl)
        while v.depth < self.maxDepth:
            nextTheta = np.random.normal(1,self.maxParamExploration,self.trajOrder)
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
        bestTheta = np.random.normal(0,self.maxParamExploration,self.trajOrder)
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
        ax = 0
        ay = 0
        bx = theta[0]
        by = theta[1]
        cx = theta[2]
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

        # lower reward if agent is out of bound
        if not methods.sanityCheck(tau[0,:],tau[1,:],v.gmrf):
            r -= 100
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
            g.append(child.totalR/(child.visits-1) + self.kappa*math.sqrt(2*math.log(v.visits)/child.visits))
        return v.children[np.argmax(g)]
