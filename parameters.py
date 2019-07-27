import math
import numpy as np

class par:
    def __init__(self, belief, control, cbtsNodeBelief, fieldType, temporal, plot, nIter, K=15, H=10, nUpdated=5, lambd=1e-1,
                 pi2ControlCost=5, branchingFactor=6, maxDepth=3, kappa=100, kappaChildSelection=1,
                 UCBRewardFactor=0.05, cbtsControlCost=0.2, discountFactor=0.5):
        self.belief = belief
        self.control = control
        self.cbtsNodeBelief = cbtsNodeBelief
        self.fieldType = fieldType
        self.temporal = temporal
        self.plot = plot
        self.nIter = nIter  # number of iterations

        self.showExploredPaths = False
        self.showActionRewardMapping = False
        self.showAcquisitionFunction = False
        self.showPerformance = False

        self.exploitingRate = 0

        self.dt = 1  # timesteps per iteration
        self.nMeas = 10  # number of measurements for bayesian inference (nMeas = nIter for inference without truncation)
        self.ov2 = 0.01  # measurement variance
        self.ov2Real = self.ov2
        self.dX = 0.01
        self.dY = 0.01  # discretizaton in x and y for Plotting

        self.x0 = 0
        self.y0 = 0
        self.alpha0 = math.pi / 4  # initial state
        self.maxStepsize = 0.5  # maximum change in every state per iteration
        self.xVel = self.maxStepsize
        self.yVel = self.maxStepsize

        self.dxdt = 0.001  # Shift of true field in x direction
        self.dydt = 0.001  # Shift of true field in y direction
        self.pulseTime = self.nIter  # Duration of sinusodial pulsation

        """GMRF class"""
        self.xMin = 0  # GMRF dimensions
        self.xMax = 10
        self.nGridX = 30
        self.yMin = 0
        self.yMax = 10
        self.nGridY = 30
        self.nBeta = 1  # regression coefficients
        self.nEdge = 5  # needs to be at least 1
        self.valueT = 1e-3  # Prior precision value for regression vector bet

        self.nGridXSampled = 10
        self.nGridYSampled = 10

        """STKF class"""
        self.sigmaT = 1e3  # 1e3    # determines exponential decay of time kernel
        self.lambdSTKF = 1  # influences time kernel value
        self.sigma2 = 0.01

        """PI2 controller"""
        "Rollout Tuning"
        self.K = K  # number of path roll outs
        self.H = H  # control horizon steps
        self.nUpdated = nUpdated  # number of iterations
        self.lambd = lambd  # 1e-2 # rescales state costs, affects noise of path roll-outs (positively)
        self.outOfGridPenaltyPI2 = 10  # each observation outside of grid adds a negative reward
        self.pi2ControlCost = pi2ControlCost  # 5e-1   # affects noise of path roll-outs (negatively)
        #ctrSamplingTime = 0.1  # time discretization

        """CBTS controller"""
        self.constantStepsize = True
        self.branchingFactor = branchingFactor  # number of actions that can be evaluated at max for each path segment
        self.maxDepth = maxDepth  # depth of search tree
        self.kappa = kappa  # large: evaluate more untried actions; small: concentrate on actions which already lead to high rewards
        self.kappaChildSelection = kappaChildSelection  # high value: expand nodes with less visits, low: expand nodes with high accumulated reward
        self.UCBRewardFactor = UCBRewardFactor  # reward = variance + UCBRewardFactor*mean
        self.outOfGridPenaltyCBTS = 1
        self.cbtsControlCost = cbtsControlCost
        self.discountFactor = discountFactor  # discounts future rewards

        self.trajStepSize = 1  # determines number of measurement points along trajectory (depends on maxStepsize)
        self.trajScaling = 1  # scales trajectories (cx and cy in case of quadratic trajectories)
        self.CBTSIterations = 20  # determines runtime of algorithm, could also be done with time limit
        self.nMeasPoints = int(self.trajStepSize / self.maxStepsize) # number of measurement points along trajectory
        self.nTrajPoints = self.nMeasPoints + 1 # length of trajectories (including starting position)

        self.thetaMin = -1  # determines curvature of generated trajectories
        self.thetaMax = 1  # determines curvature of generated trajectories
        self.thetaExpMin = self.thetaMin  # determines curvature of generated trajectories for node exploration
        self.thetaExpMax = self.thetaMax  # determines curvature of generated trajectories for node exploration
        self.trajOrder = 1  # if higher order is used check trajectory generation function
        self.initialTheta = np.zeros(self.trajOrder)  # leads to first trajectory being straight

        # Gaussian Process for action reward mapping
        self.kernelPar = 10  # used in exponential kernel to determine variance between to inputs
        self.nThetaSamples = 50  # number of samples thetas which are candidates for next theta

        """Random Walk"""
        self.noiseRandomWalk = 0.3

        if self.belief != 'regBayesTrunc':
            self.nMeas = self.nIter
        if self.belief == 'stkf':
            self.nBeta = 0