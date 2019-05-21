import math
import numpy as np
import parameters as par

def generateTrajectory(theta, pos):
    # create anchor points
    nAnchorPoints = 3
    trajectoryNoise = 0.01

    lMax = 10
    eps = 1
    delta = 1
    Traj = np.zeros((2, nAnchorPoints))

    anchorPoints = np.zeros((2, nAnchorPoints))
    anchorPoints[:, 0] = pos[:,0]
    y = np.zeros((1, nAnchorPoints))
    y[0, 0] = 0

    for i in range(nAnchorPoints - 1):
        alpha = 2 * math.pi * np.random.normal(trajectoryNoise)
        anchorPoints[0, i + 1] = anchorPoints[0, i] + par.maxStepsize / nAnchorPoints * math.cos(alpha)
        anchorPoints[1, i + 1] = anchorPoints[1, i] + par.maxStepsize / nAnchorPoints * math.sin(alpha)
        y[0, i + 1] = par.dt / nAnchorPoints
        # calculate weight vector
        GX = np.zeros((nAnchorPoints, nAnchorPoints))
        GY = np.zeros((nAnchorPoints, nAnchorPoints))
        mPi = np.zeros((nAnchorPoints, 1))

        for i in range(nAnchorPoints):
            for j in range(nAnchorPoints):
                GX[i, j] = RBFkernel(anchorPoints[:, i], anchorPoints[:, j])
                GY[i, j] = RBFkernel(y[0, i], y[0, j])
            mPi[i, 0] = sum(np.random.choice(GX.flatten(), lMax)) / lMax

        Lambd = np.dot(np.linalg.inv(GX + eps * np.eye(nAnchorPoints)), mPi)
        LambdGY = np.dot(Lambd.T, GY)
        for y in range(nAnchorPoints):
            factor1 = np.linalg.inv(np.dot(LambdGY, LambdGY.T) + eps)[0,0]
            factor2 = np.dot(LambdGY,Lambd[:,0])[0]
            w = factor1*factor2*GY[:, y].reshape(nAnchorPoints,1)
            Traj[y] = sum(np.dot(anchorPoints, w))
    return Traj


def RBFkernel(vec1, vec2):
    return math.exp(-(np.linalg.norm(vec1 - vec2) ** 2) / (2 * sigmaKernel ** 2))

sigmaKernel = 0.01
pos = np.array([[0],[0]])
Traj = generateTrajectory(0,pos)
test = 1
