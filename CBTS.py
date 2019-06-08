import math
import numpy as np
import methods
import copy
from additionalClassesCBTS import node, mapActionReward

class CBTS:
    def __init__(self, nIterations, nTrajPoints, maxParamExploration, trajOrder, maxDepth, branchingFactor, kappa):
        self.nIterations = nIterations
        self.nTrajPoints = nTrajPoints
        self.maxParamExploration = maxParamExploration
        self.trajOrder = trajOrder
        self.maxDepth = maxDepth
        self.branchingFactor = branchingFactor # maximum number of generated actions per node
        self.kappa = kappa
        self.xTraj = np.zeros((self.nTrajPoints,1))
        self.yTraj = np.zeros((self.nTrajPoints,1))

        self.map = mapActionReward(-0.5,0.5,30,2)


    def getNewTraj(self, auv, gmrf):
        v0 = node(gmrf,auv,0) # create node with belief b and total reward 0
        self.xTraj = np.zeros((self.nTrajPoints, 1))
        self.yTraj = np.zeros((self.nTrajPoints, 1))
        for i in range(self.nIterations):
            print("CBTS-Iteration",i,"of",self.nIterations)
            vl = self.treePolicy(v0) # get next node
            if vl == None:
                continue # max depth and branching reached
            r = self.exploreNode(vl)
            #print("exploring node ",vl," at position",vl.auv.x,vl.auv.y, " yields reward of",r)
            self.backUp(v0,vl,vl.totalR+r)
            #print("Level 1 nodes after backing up:")
            #for Eachnode in v0.children:
                #print("Node:",Eachnode,"/Reward: ",Eachnode.totalR,"/Counter: ",Eachnode.visits)
            #print("Best trajectory is now returned")
            #print("_______________________________")
        bestTraj, derivX, derivY = self.getBestTheta(v0)
        return bestTraj, derivX, derivY

    def treePolicy(self,v):
        #print(" call tree policy:")
        while v.depth < self.maxDepth:
            if len(v.D) == 0:
                self.map.resetMapping()
            if len(v.D) < self.branchingFactor:
                #print("     generate new node at depth ",v.depth)
                theta = self.getNextTheta()
                #print(theta)
                traj, derivX, derivY = self.generateTrajectory(v, theta)
                self.xTraj = np.hstack((self.xTraj,traj[0,:].reshape(self.nTrajPoints,1)))
                self.yTraj = np.hstack((self.yTraj,traj[1,:].reshape(self.nTrajPoints,1)))

                r,o = self.evaluateTrajectory(v,traj)
                v.D.append((theta,r))

                #print("     generated trajectory: ",traj)
                #print("     with theta = ",theta)
                #print("     data set is now: ",v.D)
                #print("     reward is: ",r)

                # Update GP mapping from theta to r:
                self.map.updateMapActionReward(theta,r)
                #print("Update GP mapping:")
                #print("Theta:",theta)
                #print("Reward:",r)
                #print(np.max(self.map.meanCond))
                #print("___")

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
                vNew.auv.derivX = derivX
                vNew.auv.derivY = derivY
                return vNew
            else:
                #print("No more actions. Switching from node",v)
                v = self.bestChild(v)
                #print("to node",v)

    def getNextTheta(self):
        b = self.map.meanCond + self.kappa * self.map.covCond.diagonal().reshape(self.map.nGridPoints, 1)
        index = np.random.choice(np.flatnonzero(b == b.max()))
        bestTheta = self.map.convertIndextoTheta(index)
        #print("getNextTheta:")
        #print(bestTheta)
        print("Index of best theta:", index)
        return bestTheta

    def exploreNode(self,vl):
        r = 0
        v = copy.deepcopy(vl)
        while v.depth < self.maxDepth:
            nextTheta = np.random.normal(1,self.maxParamExploration,self.trajOrder)
            nextTheta = np.expand_dims(nextTheta, axis=0)
            nextTraj, derivX, derivY = self.generateTrajectory(v,nextTheta)
            dr,do = self.evaluateTrajectory(v,nextTraj)
            r += dr
            v.auv.x = nextTraj[0,-1]
            v.auv.y = nextTraj[1,-1]
            v.auv.derivX = derivX
            v.auv.derivY = derivY
            v.depth += 1
        return r

    def getBestTheta(self,v0):
        maxR = -math.inf
        bestTheta = np.random.normal(0,self.maxParamExploration,self.trajOrder)
        bestTheta = np.expand_dims(bestTheta, axis=0)
        for theta,r in v0.D:
            if r > maxR:
                bestTheta = theta
                maxR = r
        bestTraj, derivX, derivY = self.generateTrajectory(v0,bestTheta)
        print(v0.D)
        return bestTraj, derivX, derivY

    def generateTrajectory(self,v, theta):
        # theta = [ax ay bx by cx]
        # beta =    [dx cx bx ax]
        #           [dy cy by ay]
        # dx = posX, dy = posY, cx = dC1/du|u=0 = derivX, cy = dC2/du|u=0 = derivY
        ax = 0
        ay = 0
        bx = theta[0, 0]
        by = theta[0, 1]
        cx = v.auv.derivX
        cy = v.auv.derivY
        dx = v.auv.x
        dy = v.auv.y

        beta = np.array([[dx, cx, bx, ax], [dy, cy, by, ay]])

        tau = np.zeros((2,self.nTrajPoints))
        for i in range(self.nTrajPoints):
            u = i / (self.nTrajPoints - 1)
            tau[:, i] = np.dot(beta, np.array([[1], [u], [u ** 2], [u ** 3]]))[:, 0]

        derivX = 3 * ax + 2 * bx + cx
        derivY = 3 * ay + 2 * by + cy

        return tau, derivX, derivY

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
