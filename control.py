import copy
import gc
import math

import numpy as np

import functions
from classes import node


class piControl:
    def __init__(self, par):
        self.lambd = par.lambd  # influences state costs and noise variance
        self.H = par.H  # control horizon steps
        self.g = np.eye(1)
        self.K = par.K  # number of path roll outs
        self.dt = par.dt  # time discretization
        self.nUpdated = par.nUpdated  # number of iterations
        self.u = np.zeros((self.H, 1))
        self.outOfGridPenaltyPI2 = par.outOfGridPenaltyPI2
        self.pi2ControlCost = par.pi2ControlCost
        self.R = self.pi2ControlCost * np.eye(1)  # input cost matrix
        try:
            self.varNoise = self.lambd * np.linalg.inv(self.R)
        except:
            self.varNoise = None

        self.xTraj = np.zeros((1, self.K))
        self.yTraj = np.zeros((1, self.K))
        self.alphaTraj = np.zeros((1, self.K))
        self.xPathRollOut = np.zeros((1, self.K))
        self.yPathRollOut = np.zeros((1, self.K))

    def getNewState(self, auv, gmrf):
        self.u[0:-2, 0] = self.u[1:-1, 0]
        self.u[-1, 0] = 0

        # M = np.dot(np.linalg.inv(self.R), np.dot(self.g, self.g.T)) / (
        #    np.dot(self.g.T, np.dot(np.linalg.inv(self.R), self.g)))
        # If input is one dimensional and noise directly affects input:
        M = 1

        noise = np.zeros((self.H, self.K))
        self.xPathRollOut = np.zeros((self.H, self.K))
        self.yPathRollOut = np.zeros((self.H, self.K))
        S = np.zeros((self.H, self.K))
        expS = np.zeros((self.H, self.K))
        P = np.zeros((self.H, self.K))
        stateCost = np.zeros((self.H, self.K))
        controlCost = np.zeros((self.H, self.K))

        for n in range(self.nUpdated):

            for k in range(self.K):
                # sample control noise and compute path roll-outs
                # noise[:, k] = math.sqrt(self.varNoise) * np.random.standard_normal(self.H)
                noise[:, k] = math.pi / 16 * np.random.standard_normal(self.H)
                (xTrVec, yTrVec, alphaNew) = auv.trajectoryFromControl(self.u[:, 0] + noise[:, k])
                self.xPathRollOut[:, k] = xTrVec[:, 0]
                self.yPathRollOut[:, k] = yTrVec[:, 0]

                # compute path costs
                for h in range(self.H):
                    if not functions.sanityCheck(self.xPathRollOut[h, k] * np.eye(1),
                                                 self.yPathRollOut[h, k] * np.eye(1), gmrf):
                        stateCost[h, k] = self.outOfGridPenaltyPI2
                        controlCost[h, k] = 0
                    else:
                        Phi = functions.mapConDis(gmrf, self.xPathRollOut[h, k], self.yPathRollOut[h, k])
                        stateCost[h, k] = 1 / np.dot(Phi, gmrf.diagCovCond)
                        uHead = self.u[h, 0] + M * noise[h, k]
                        controlCost[h, k] = 0.5 * np.dot(uHead.T, np.dot(self.R, uHead))

                for h in range(self.H):
                    S[h, k] = np.sum(stateCost[h:, k]) + np.sum(controlCost[h:, k])

                for h in range(self.H):
                    expS[h, k] = math.exp(
                        -((S[h, k] - np.amin(S[:, k])) / (np.amax(S[:, k]) - np.amin(S[:, k]))) / self.lambd)

            deltaU = np.zeros((self.H, 1))
            for h in range(self.H):
                for k in range(self.K):
                    P[h, k] = expS[h, k] / np.sum(expS[h, :])

                for k in range(self.K):
                    deltaU[h] += P[h, k] * M * noise[h, k]
            self.u += deltaU

        self.xTraj, self.yTraj, self.alphaTraj = auv.trajectoryFromControl(self.u)

        xNext, yNext, alphaNext = (self.xTraj[1], self.yTraj[1], self.alphaTraj[1])

        if xNext < gmrf.xMin:
            xNext = gmrf.xMin
            if math.pi/2 < alphaNext < math.pi:
                alphaNext = 0.49 * math.pi
            if math.pi < alphaNext < 1.5 * math.pi:
                alphaNext = 1.51 * math.pi

        if xNext > gmrf.xMax:
            xNext = gmrf.xMax
            if 0 < alphaNext < 0.5 * math.pi:
                alphaNext = 0.51 * math.pi
            if 1.5 * math.pi < alphaNext < 2 * math.pi:
                    alphaNext = 1.49 * math.pi

        if yNext < gmrf.yMin:
            yNext = gmrf.yMin
            if math.pi < alphaNext < 1.49*math.pi:
                alphaNext = math.pi
            if 1.5*math.pi < alphaNext < 2*math.pi:
                alphaNext = 0.01

        if yNext > gmrf.yMax:
            yNext = gmrf.yMax
            if 0 < alphaNext < 0.5 * math.pi:
                alphaNext = 1.99 * math.pi
            if 0.5 * math.pi < alphaNext < math.pi:
                alphaNext = 1.01 * math.pi

        return xNext, yNext, alphaNext


class CBTS:
    def __init__(self, par):
        self.par = par
        self.nIterations = par.CBTSIterations
        self.nTrajPoints = par.nTrajPoints
        self.trajOrder = par.trajOrder
        self.maxDepth = par.maxDepth
        self.branchingFactor = par.branchingFactor  # maximum number of generated actions per node
        self.kappa = par.kappa
        self.discountFactor = par.discountFactor
        self.cbtsControlCost = par.cbtsControlCost
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

    def getNewTraj(self, auv, gmrf):
        # Get gmrf with less grid points
        v0 = node(self.par, copy.deepcopy(gmrf), auv)  # create node with belief b and total reward 0
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
        gc.collect()

        for ix in range(self.nTrajPoints):
            if bestTraj[0, ix] < gmrf.xMin:
                bestTraj[0, ix] = gmrf.xMin
            elif bestTraj[0, ix] > gmrf.xMax:
                bestTraj[0, ix] = gmrf.xMax
        for iy in range(self.nTrajPoints):
            if bestTraj[1, iy] < gmrf.yMin:
                bestTraj[1, iy] = gmrf.yMin
            elif bestTraj[1, iy] > gmrf.yMax:
                bestTraj[1, iy] = gmrf.yMax
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
                if self.par.cbtsNodeBelief == 'noUpdates':
                    vNewGMRF = v.gmrf  # pass reference (since no updates)
                elif self.par.cbtsNodeBelief == 'sampledGMRF' and v.depth == 0:
                    vNewGMRF = functions.sampleGMRF(v.gmrf)
                else:
                    vNewGMRF = copy.deepcopy(v.gmrf)  # copy GMRF (since copy will be updated)

                vNew = node(self.par, vNewGMRF, v.auv)
                vNew.rewardToNode = v.rewardToNode + self.discountFactor ** v.depth * r
                vNew.totalReward = vNew.rewardToNode
                vNew.parent = v
                vNew.depth = v.depth + 1
                vNew.actionToNode = theta

                v.children.append(vNew)

                # simulate GP update of belief
                for i in range(len(o)):
                    vNew.auv.x = traj[0, i + 1]
                    vNew.auv.y = traj[1, i + 1]
                    if self.par.cbtsNodeBelief != 'noUpdates':
                        Phi = functions.mapConDis(vNew.gmrf, vNew.auv.x, vNew.auv.y)
                        vNew.gmrf.seqBayesianUpdate(o[i], Phi)
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
            if self.par.showActionRewardMapping and len(v.D) == (self.branchingFactor - 1):
                functions.plotPolicy(self.par, v.GP, thetaPredict, mu)

        return bestTheta

    def exploreNode(self, vl):
        r = 0
        v = copy.deepcopy(vl)
        while v.depth < self.maxDepth:
            nextTheta = np.random.uniform(self.thetaExpMin, self.thetaExpMax, self.trajOrder)
            nextTraj, derivX, derivY = self.generateTrajectory(v, nextTheta)

            # add explored paths to collected trajectories for plotting:
            if self.par.showExploredPaths:
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
        if self.par.showAcquisitionFunction:
            functions.plotRewardFunction(self.par, v0.gmrf)

        return bestTraj, derivX, derivY

    def generateTrajectory(self, v, theta):
        # theta = [ax ay bx by cx]
        # beta =    [dx cx bx ax]
        #           [dy cy by ay]
        # dx = posX, dy = posY, cx = dC1/du|u=0 = derivX, cy = dC2/du|u=0 = derivY
        ax = 0
        ay = 0
        bx = 0
        by = 0

        if self.trajOrder == 1:
            if v.auv.derivX > 0:
                if v.auv.derivY > 0:
                    if theta[0] > 0:
                        bx = 0
                        by = -theta[0]
                    elif theta[0] < 0:
                        bx = theta[0]
                        by = 0
                elif v.auv.derivY < 0:
                    if theta[0] > 0:
                        bx = -theta[0]
                        by = 0
                    elif theta[0] < 0:
                        bx = 0
                        by = -theta[0]
            elif v.auv.derivX < 0:
                if v.auv.derivY > 0:
                    if theta[0] > 0:
                        bx = theta[0]
                        by = 0
                    elif theta[0] < 0:
                        bx = 0
                        by = theta[0]
                elif v.auv.derivY < 0:
                    if theta[0] > 0:
                        bx = 0
                        by = theta[0]
                    elif theta[0] < 0:
                        bx = -theta[0]
                        by = 0
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
        u = 0
        length = 0
        discretize = 1 / 1000
        for i in range(self.nTrajPoints):
            tau[:, i] = np.dot(beta, np.array([[1], [u], [u ** 2], [u ** 3]]))[:, 0]

            if self.par.constantStepsize:
                # rescaled u in order to generate steps with a fixed length
                while length < self.par.maxStepsize:
                    u += discretize
                    length += discretize * math.sqrt((2 * by * u + cy) ** 2 + (2 * bx * u + cx) ** 2)
                u -= discretize
                length = 0
            else:
                u = (i + 1) / self.nTrajPoints

        derivX = 3 * ax * u**2 + 2 * bx * u + cx
        derivY = 3 * ay * u**2 + 2 * by * u + cy

        return tau, derivX, derivY

    def evaluateTrajectory(self, v, tau, theta):
        r = - self.cbtsControlCost * abs(theta)
        o = []
        for i in range(self.nTrajPoints - 1):
            Phi = functions.mapConDis(v.gmrf, tau[0, i + 1], tau[1, i + 1])
            r += (np.dot(Phi, v.gmrf.diagCovCond / max(v.gmrf.diagCovCond)) + self.UCBRewardFactor * np.dot(Phi,
                                                                           v.gmrf.meanCond / max(v.gmrf.meanCond)))[0]
            o.append(np.dot(Phi, v.gmrf.meanCond))
            # lower reward if agent is out of bound
            if not functions.sanityCheck(tau[0, i + 1] * np.eye(1), tau[1, i + 1] * np.eye(1), v.gmrf):
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
