import numpy as np
import math
import parameters as par
import methods
import copy
from classes import node
import matplotlib.pyplot as plt

class piControl:
    def __init__(self, R, g, lambd, H, K, dt, nUpdated):
        self.R = R  # input cost matrix
        self.g = g  # mapping from u to states
        self.lambd = lambd  # influences state costs and noise variance
        self.varNoise = self.lambd*np.linalg.inv(self.R)
        self.H = H  # control horizon steps
        self.K = K  # number of path roll outs
        self.dt = dt    # time discretization
        self.nUpdated = nUpdated    # number of iterations
        self.u = np.zeros((self.H, 1))

        self.xTraj = np.zeros((1, self.K))
        self.yTraj = np.zeros((1, self.K))
        self.alphaTraj = np.zeros((1, self.K))
        self.xPathRollOut = np.zeros((1, self.K))
        self.yPathRollOut = np.zeros((1, self.K))

    def getNewState(self, auv, gmrf):
        self.u[0:-2,0] = self.u[1:-1,0]
        self.u[-1,0] = 0

        M = np.dot(np.linalg.inv(self.R), np.dot(self.g, self.g.T))/(np.dot(self.g.T, np.dot(np.linalg.inv(self.R), self.g)))
        for n in range(self.nUpdated):
            noise = np.zeros((self.H, self.K))
            self.xPathRollOut = np.zeros((self.H, self.K))
            self.yPathRollOut = np.zeros((self.H, self.K))
            S = np.zeros((self.H + 1, self.K))

            for k in range(self.K):

                # sample control noise and compute path roll-outs
                for j in range(self.H):
                    noise[:, k] = np.random.normal(0, math.sqrt(self.varNoise[j, j]))

                (xTrVec, yTrVec, alphaNew) = auv.trajectoryFromControl(self.u[:, 0] + noise[:, k])
                self.xPathRollOut[:, k] = xTrVec[:, 0]
                self.yPathRollOut[:, k] = yTrVec[:, 0]

                # compute path costs
                stateCost = 0
                for i in range(self.H):
                    index = self.H-i-1
                    if not methods.sanityCheck(self.xPathRollOut[index, k]*np.eye(1), self.yPathRollOut[index, k]*np.eye(1), gmrf):
                        stateCost += par.outOfGridPenalty
                    else:
                        Phi = methods.mapConDis(gmrf, self.xPathRollOut[index, k], self.yPathRollOut[index, k])
                        stateCost += 1/np.dot(Phi,gmrf.diagCovCond)
                    uHead = self.u[index:self.H,0] + np.dot(M[index:self.H,index:self.H],noise[index:self.H,k])
                    S[index, k] = S[index+1, k] + stateCost + 0.5*np.dot(uHead.T, np.dot(self.R[index:self.H,index:self.H],uHead))

            # Normalize state costs
            S = S/np.amax(S)

            # Compute cost of path segments
            expS = np.zeros((self.H, self.K))
            for k in range(self.K):
                for i in range(self.H):
                    expS[i,k] = math.exp(-S[i, k]/self.lambd)

            P = np.zeros((self.H, self.K))
            for k in range(self.K):
                for i in range(self.H):
                    P[i, k] = expS[i, k] / sum(expS[i,:])

            # Compute next control action
            deltaU = np.zeros((self.H, self.H))
            weightedDeltaU = np.zeros((self.H, 1))
            for i in range(self.H):
                deltaU[i:self.H, i] = np.dot(np.dot(M[i:self.H, i:self.H], noise[i:self.H,:]),P[i,:].T)
                sumNum = 0
                sumDen = 0
                for h in range(self.H):
                    sumNum += (self.H - h)*deltaU[:, i][i]
                    sumDen += (self.H - h)
                weightedDeltaU[i, 0] = sumNum/sumDen

            self.u += weightedDeltaU

        self.xTraj, self.yTraj, self.alphaTraj = auv.trajectoryFromControl(self.u)

        auv.x, auv.y, auv.alpha = (self.xTraj[1], self.yTraj[1], self.alphaTraj[1])

        if auv.x < gmrf.xMin:
            auv.x = gmrf.xMin
        elif auv.x > gmrf.xMax:
            auv.x = gmrf.xMax

        if auv.y < gmrf.yMin:
            auv.y = gmrf.yMin
        elif auv.y > gmrf.yMax:
            auv.y = gmrf.yMax

        return auv.x, auv.y

class CBTS:
    def __init__(self, nIterations, nTrajPoints, trajOrder, maxDepth, branchingFactor, kappa):
        self.nIterations = nIterations
        self.nTrajPoints = nTrajPoints
        self.trajOrder = trajOrder
        self.maxDepth = maxDepth
        self.branchingFactor = branchingFactor # maximum number of generated actions per node
        self.kappa = kappa
        self.xTraj = np.zeros((self.nTrajPoints,1))
        self.yTraj = np.zeros((self.nTrajPoints,1))

    def getNewTraj(self, auv, gmrf):
        v0 = node(gmrf,auv) # create node with belief b and total reward 0
        self.xTraj = np.zeros((self.nTrajPoints, 1))
        self.yTraj = np.zeros((self.nTrajPoints, 1))
        for i in range(self.nIterations):
            print("CBTS-Iteration",i,"of",self.nIterations)
            vl = self.treePolicy(v0) # get next node
            if vl is None:
                continue # max depth and branching reached

            # rollout
            futureReward = self.exploreNode(vl)
            vl.accReward = vl.rewardToNode + futureReward
            print("exploring node ",vl," at position",vl.auv.x,vl.auv.y, " yields reward of",futureReward)
            self.backUp(v0,vl,vl.accReward)
            print("Level 1 nodes after backing up:")
            for Eachnode in v0.children:
                print("Node:",Eachnode,"/Reward: ",Eachnode.accReward,"/Counter: ",Eachnode.visits)
        print("Best trajectory is now returned")
        print("_______________________________")
        bestTraj, derivX, derivY = self.getBestTheta(v0)
        return bestTraj, derivX, derivY

    def treePolicy(self,v):
        print(" call tree policy:")
        while v.depth < self.maxDepth:
            print(len(v.D)," nodes have been created at this depth.")
            if len(v.D) < self.branchingFactor:
                print("     generate new node at depth ",v.depth)
                theta = self.getNextTheta(v)
                traj, derivX, derivY = self.generateTrajectory(v, theta)
                self.xTraj = np.hstack((self.xTraj,traj[0,:].reshape(self.nTrajPoints,1)))
                self.yTraj = np.hstack((self.yTraj,traj[1,:].reshape(self.nTrajPoints,1)))

                r,o = self.evaluateTrajectory(v,traj)
                v.D.append((theta,r))

                print("     generated trajectory: ",traj)
                print("     with theta = ",theta)
                print("     data set is now: ",v.D)
                print("     reward is: ",r)

                # Update GP mapping from theta to r:
                v.GP.update(theta,r)

                # Create new node:
                vNew = node(v.gmrf,v.auv)
                vNew.rewardToNode = v.rewardToNode + par.discountFactor**v.depth * r
                vNew.totalReward = vNew.rewardToNode
                vNew.parent = v
                vNew.depth = v.depth + 1
                vNew.actionToNode = theta

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
                print("No more actions at node",v)
                v = self.bestChild(v)
                print("switching to node",v)


    def getNextTheta(self,v):
        if v.GP.emptyData:
            bestTheta = par.initialTheta
            #bestTheta = np.random.uniform(par.thetaMin,par.thetaMax,par.trajOrder)
        else:
            thetaPredict = np.random.uniform(par.thetaMin,par.thetaMax,(par.nThetaSamples,par.trajOrder))
            mu,var = v.GP.predict(thetaPredict)
            h = mu + self.kappa * var.diagonal().reshape(par.nThetaSamples,1)
            index = np.argmax(h)
            bestTheta = thetaPredict[index,:]
            print("getNextTheta:")
            print(bestTheta)
            print("Index of best theta:", index)

        # add next theta to GP

        # plot estimated reward over actions
        if par.plotOptions.showActionRewardMapping and len(v.D) == (self.branchingFactor-1):
            methods.plotPolicy(v.GP,thetaPredict,mu)

        return bestTheta

    def exploreNode(self,vl):
        r = 0
        v = copy.deepcopy(vl)
        counter = 0
        while v.depth < self.maxDepth:
            nextTheta = np.random.uniform(par.thetaMin,par.thetaMax,par.trajOrder)
            nextTraj, derivX, derivY = self.generateTrajectory(v,nextTheta)

            # add explored paths to collected trajectories for plotting:
            if par.plotOptions.showExploredPaths:
                self.xTraj = np.hstack((self.xTraj, nextTraj[0, :].reshape(self.nTrajPoints, 1)))
                self.yTraj = np.hstack((self.yTraj, nextTraj[1, :].reshape(self.nTrajPoints, 1)))

            dr,do = self.evaluateTrajectory(v,nextTraj)
            r += par.discountFactor**v.depth * dr
            v.auv.x = nextTraj[0,-1]
            v.auv.y = nextTraj[1,-1]
            v.auv.derivX = derivX
            v.auv.derivY = derivY
            v.depth += 1
            counter += 1
        if counter > 0:
            return r/counter
        else:
            return r

    def getBestTheta(self,v0):
        maxR = -math.inf
        bestTheta = np.zeros(2)
        for child in v0.children:
            if child.accReward > maxR:
                bestTheta = child.actionToNode
                maxR = child.accReward
        bestTraj, derivX, derivY = self.generateTrajectory(v0,bestTheta)
        print("chosen best theta: ",bestTheta," with trajectory ",bestTraj)
        return bestTraj, derivX, derivY

    def generateTrajectory(self,v, theta):
        # theta = [ax ay bx by cx]
        # beta =    [dx cx bx ax]
        #           [dy cy by ay]
        # dx = posX, dy = posY, cx = dC1/du|u=0 = derivX, cy = dC2/du|u=0 = derivY
        ax = 0
        ay = 0

        if par.trajOrder == 1:
            if theta[0] < 0:
                bx = theta[0]
                by = 0
            elif theta[0] >= 0:
                bx = 0
                by = -theta[0]
        if par.trajOrder == 2:
            bx = theta[0]
            by = theta[1]
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
            r -= par.outOfGridPenalty
        return r,o

    def backUp(self,v0,v,reward):
        v = v.parent
        while v != v0:
            v.visits += 1
            v.accReward += reward
            v = v.parent

    def bestChild(self,v):
        g = []
        for child in v.children:
            g.append(child.accReward/child.visits + par.kappaChildSelection*math.sqrt(2*math.log(v.visits)/child.visits))
        return v.children[np.argmax(g)]
