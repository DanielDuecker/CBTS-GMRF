import copy

import math
import numpy as np

import functions
from classes import node
from classes import gmrf


class piControl:
    def __init__(self,par):
        self.R = par.R  # input cost matrix
        self.g = par.g  # mapping from u to states
        self.lambd = par.lambd  # influences state costs and noise variance
        self.varNoise = self.lambd * np.linalg.inv(self.R)
        self.H = par.H  # control horizon steps
        self.K = par.K  # number of path roll outs
        self.dt = par.dt  # time discretization
        self.nUpdated = par.nUpdated  # number of iterations
        self.u = np.zeros((self.H, 1))
        self.outOfGridPenaltyPI2 = par.outOfGridPenaltyPI2

        self.xTraj = np.zeros((1, self.K))
        self.yTraj = np.zeros((1, self.K))
        self.alphaTraj = np.zeros((1, self.K))
        self.xPathRollOut = np.zeros((1, self.K))
        self.yPathRollOut = np.zeros((1, self.K))

    def getNewState(self, auv, gmrf):
        self.u[0:-2, 0] = self.u[1:-1, 0]
        self.u[-1, 0] = 0

        M = np.dot(np.linalg.inv(self.R), np.dot(self.g, self.g.T)) / (
            np.dot(self.g.T, np.dot(np.linalg.inv(self.R), self.g)))
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
                    index = self.H - i - 1
                    if not functions.sanityCheck(self.xPathRollOut[index, k] * np.eye(1),
                                               self.yPathRollOut[index, k] * np.eye(1), gmrf):
                        stateCost += self.outOfGridPenaltyPI2
                    else:
                        Phi = functions.mapConDis(gmrf, self.xPathRollOut[index, k], self.yPathRollOut[index, k])
                        stateCost += 1 / np.dot(Phi, gmrf.diagCovCond)
                    uHead = self.u[index:self.H, 0] + np.dot(M[index:self.H, index:self.H], noise[index:self.H, k])
                    S[index, k] = S[index + 1, k] + stateCost + 0.5 * np.dot(uHead.T,
                                                                np.dot(self.R[index:self.H, index:self.H],uHead))

            # Normalize state costs
            S = S / np.amax(S)

            # Compute cost of path segments
            expS = np.zeros((self.H, self.K))
            for k in range(self.K):
                for i in range(self.H):
                    expS[i, k] = math.exp(-S[i, k] / self.lambd)

            P = np.zeros((self.H, self.K))
            for k in range(self.K):
                for i in range(self.H):
                    P[i, k] = expS[i, k] / sum(expS[i, :])

            # Compute next control action
            deltaU = np.zeros((self.H, self.H))
            weightedDeltaU = np.zeros((self.H, 1))
            for i in range(self.H):
                deltaU[i:self.H, i] = np.dot(np.dot(M[i:self.H, i:self.H], noise[i:self.H, :]), P[i, :].T)
                sumNum = 0
                sumDen = 0
                for h in range(self.H):
                    sumNum += (self.H - h) * deltaU[:, i][i]
                    sumDen += (self.H - h)
                weightedDeltaU[i, 0] = sumNum / sumDen

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
    def __init__(self,par):
        self.par = par
        self.nIterations = par.CBTSIterations
        self.nTrajPoints = par.nTrajPoints
        self.trajOrder = par.trajOrder
        self.maxDepth = par.maxDepth
        self.branchingFactor = par.branchingFactor  # maximum number of generated actions per node
        self.kappa = par.kappa
        self.discountFactor = par.discountFactor
        self.controlCost = par.controlCost
        self.initialTheta = par.initialTheta
        self.thetaMin = par.thetaMin
        self.thetaMax = par.thetaMax
        self.thetaExpMin = par.thetaExpMin
        self.thetaExpMax = par.thetaExpMax
        self.nThetaSamples = par.nThetaSamples
        self.UCBRewardFactor = par.UCBRewardFactor
        self.kappaChildSelection = par.kappaChildSelection
        self.outOfGridPenaltyCBTS = par.outOfGridPenaltyCBTS

        self.xTraj = np.zeros((self.nTrajPoints, 1))
        self.yTraj = np.zeros((self.nTrajPoints, 1))

    def getNewTraj(self, auv, gmrfOrig):
        gmrfRed = gmrf(self.par,self.par.nGridXred,self.par.nGridYred)
        functions.sampleGMRF(gmrfOrig,gmrfRed)

        # Get gmrf with less grid points
        print("calculating..")
        v0 = node(self.par, gmrfRed, auv)  # create node with belief b and total reward 0
        self.xTraj = np.zeros((self.nTrajPoints, 1))
        self.yTraj = np.zeros((self.nTrajPoints, 1))
        for i in range(self.nIterations):
            vl = self.treePolicy(v0)  # get next node
            if vl is None:
                continue  # max depth and branching reached
            # rollout
            futureReward = self.exploreNode(vl)
            vl.accReward = vl.rewardToNode + futureReward
            self.backUp(v0, vl, vl.accReward)
        bestTraj, derivX, derivY = self.getBestTheta(v0)
        return bestTraj, derivX, derivY

    def treePolicy(self, v):
        while v.depth < self.maxDepth:
            if len(v.D) < self.branchingFactor:
                theta = self.getNextTheta(v)
                traj, derivX, derivY = self.generateTrajectory(v, theta)
                self.xTraj = np.hstack((self.xTraj, traj[0, :].reshape(self.nTrajPoints, 1)))
                self.yTraj = np.hstack((self.yTraj, traj[1, :].reshape(self.nTrajPoints, 1)))

                r, o = self.evaluateTrajectory(v, traj, theta)
                v.D.append((theta, r))

                # Update GP mapping from theta to r:
                v.GP.update(theta, r)

                # Create new node:
                vNew = node(self.par, v.gmrfRed, v.auv)
                vNew.rewardToNode = v.rewardToNode + self.discountFactor ** v.depth * r
                vNew.totalReward = vNew.rewardToNode
                vNew.parent = v
                vNew.depth = v.depth + 1
                vNew.actionToNode = theta

                v.children.append(vNew)

                # simulate regular GP update:

                # simulate GP update of belief
                for i in range(len(o)):
                    vNew.auv.x = traj[0, i + 1]
                    vNew.auv.y = traj[1, i + 1]
                    Phi = functions.mapConDis(vNew.gmrfRed, vNew.auv.x, vNew.auv.y)
                    vNew.gmrfRed.seqBayesianUpdate(o[i], Phi)

                vNew.auv.derivX = derivX
                vNew.auv.derivY = derivY
                return vNew
            else:
                v = self.bestChild(v)

    def getNextTheta(self, v):
        if v.GP.emptyData:
            bestTheta = self.initialTheta
            # bestTheta = np.random.uniform(self.thetaMin,self.thetaMax,self.trajOrder)
        else:
            thetaPredict = np.random.uniform(self.thetaMin, self.thetaMax, (self.nThetaSamples, self.trajOrder))
            mu, var = v.GP.predict(thetaPredict)
            h = mu + self.kappa * var.diagonal().reshape(self.nThetaSamples, 1)
            index = np.argmax(h)
            bestTheta = thetaPredict[index, :]

            # plot estimated reward over actions
            if self.par.plotOptions.showActionRewardMapping and len(v.D) == (self.branchingFactor - 1):
                functions.plotPolicy(self.par, v.GP, thetaPredict, mu)

        return bestTheta

    def exploreNode(self, vl):
        r = 0
        v = copy.deepcopy(vl)
        while v.depth < self.maxDepth:
            nextTheta = np.random.uniform(self.thetaExpMin, self.thetaExpMax, self.trajOrder)
            nextTraj, derivX, derivY = self.generateTrajectory(v, nextTheta)

            # add explored paths to collected trajectories for plotting:
            if self.par.plotOptions.showExploredPaths:
                self.xTraj = np.hstack((self.xTraj, nextTraj[0, :].reshape(self.nTrajPoints, 1)))
                self.yTraj = np.hstack((self.yTraj, nextTraj[1, :].reshape(self.nTrajPoints, 1)))

            dr, do = self.evaluateTrajectory(v, nextTraj, nextTheta)
            r += self.discountFactor ** v.depth * dr
            v.auv.x = nextTraj[0, -1]
            v.auv.y = nextTraj[1, -1]
            v.auv.derivX = derivX
            v.auv.derivY = derivY
            v.depth += 1
        return r

    def getBestTheta(self, v0):
        maxR = -math.inf
        bestTheta = np.zeros(2)
        for child in v0.children:
            if child.accReward > maxR:
                bestTheta = child.actionToNode
                maxR = child.accReward
        bestTraj, derivX, derivY = self.generateTrajectory(v0, bestTheta)

        # plot acquisition function
        if self.par.plotOptions.showAcquisitionFunction:
            functions.plotRewardFunction(self.par,v0.gmrfRed)

        return bestTraj, derivX, derivY

    def generateTrajectory(self, v, theta):
        # theta = [ax ay bx by cx]
        # beta =    [dx cx bx ax]
        #           [dy cy by ay]
        # dx = posX, dy = posY, cx = dC1/du|u=0 = derivX, cy = dC2/du|u=0 = derivY
        ax = 0
        ay = 0

        if self.trajOrder == 1:
            if theta[0] < 0:
                bx = np.sign(v.auv.derivX) * theta[0]
                by = 0
            elif theta[0] >= 0:
                bx = 0
                by = -np.sign(v.auv.derivY) * theta[0]
        elif self.trajOrder == 2:
            bx = theta[0]
            by = theta[1]
        else:
            bx = 0
            by = 0
        cx = v.auv.derivX
        cy = v.auv.derivY
        dx = v.auv.x
        dy = v.auv.y

        beta = np.array([[dx, cx, bx, ax], [dy, cy, by, ay]])

        tau = np.zeros((2, self.nTrajPoints))
        for i in range(self.nTrajPoints):
            u = i / (self.nTrajPoints - 1)
            tau[:, i] = np.dot(beta, np.array([[1], [u], [u ** 2], [u ** 3]]))[:, 0]

        derivX = 3 * ax + 2 * bx + cx
        derivY = 3 * ay + 2 * by + cy

        return tau, derivX, derivY

    def evaluateTrajectory(self, v, tau, theta):
        r = - self.controlCost * theta
        o = []
        for i in range(self.nTrajPoints - 1):
            Phi = functions.mapConDis(v.gmrfRed, tau[0, i + 1], tau[1, i + 1])
            r += (np.dot(Phi, v.gmrfRed.covCond.diagonal()) + self.UCBRewardFactor * np.dot(Phi, v.gmrfRed.meanCond))[0]
            o.append(np.dot(Phi, v.gmrfRed.meanCond))
            # lower reward if agent is out of bound
            if not functions.sanityCheck(tau[0, i + 1] * np.eye(1), tau[1, i + 1] * np.eye(1), v.gmrfRed):
                r -= self.outOfGridPenaltyCBTS

        return r, o

    def backUp(self, v0, v, reward):
        v = v.parent
        while v != v0:
            v.visits += 1
            v.accReward += reward
            v = v.parent

    def bestChild(self, v):
        g = []
        for child in v.children:
            g.append(child.accReward / child.visits + self.kappaChildSelection * math.sqrt(
                2 * math.log(v.visits) / child.visits))
        return v.children[np.argmax(g)]
