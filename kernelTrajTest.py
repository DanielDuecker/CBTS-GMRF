import math
import numpy as np
import parameters as par
import matplotlib.pyplot as plt

def generateTrajectory(theta, pos):
    # create anchor points
    nAnchorPoints = 5
    trajectoryNoise = 0.05
    maxStepsize = 1

    lMax = 10
    eps = 1
    delta = 1

    anchorPoints = np.zeros((2, nAnchorPoints))
    anchorPoints[:, 0] = pos[:,0]
    y = np.zeros((1, nAnchorPoints))
    y[0, 0] = 0
    alpha = math.pi/4
    for i in range(nAnchorPoints - 1):
        alpha = alpha + 2*math.pi * np.random.normal(0,trajectoryNoise)
        anchorPoints[0, i + 1] = anchorPoints[0, i] + maxStepsize / nAnchorPoints * math.cos(alpha)
        anchorPoints[1, i + 1] = anchorPoints[1, i] + maxStepsize / nAnchorPoints * math.sin(alpha)
        y[0, i + 1] = y[0, i] + par.dt / nAnchorPoints
        # calculate weight vector
    GX = np.zeros((nAnchorPoints, nAnchorPoints))
    GY = np.zeros((nAnchorPoints, nAnchorPoints))

    plt.figure()
    plt.plot(anchorPoints[0, :], anchorPoints[1, :])
    plt.title("True field")
    plt.show()

    mPi = np.zeros((nAnchorPoints, nAnchorPoints))
    for i in range(nAnchorPoints):
        for j in range(nAnchorPoints):
            GX[i, j] = RBFkernel(anchorPoints[:, i], anchorPoints[:, j])
            GY[i, j] = RBFkernel(y[0, i], y[0, j])

    GXTriang = np.triu(GX)
    for i in range(nAnchorPoints):
        for j in range(nAnchorPoints):
            mPi[i, j] = sum(np.random.choice(GXTriang[GXTriang != 0], lMax)) / lMax

    Traj = np.zeros((2,nAnchorPoints))
    for iy in range(nAnchorPoints):
        Lambd = np.dot(np.linalg.inv(GX + nAnchorPoints * eps * np.eye(nAnchorPoints)), mPi[:,iy])
        LambdGY = np.dot(Lambd.T, GY)
        factor1 = 1/(np.dot(LambdGY, LambdGY.T) + delta)
        factor2 = 1/np.dot(LambdGY,Lambd)
        wy = factor1*factor2*GY[iy,:].reshape(nAnchorPoints,1)
        Traj[:,iy] = np.dot(anchorPoints, wy)[:,0]
    return Traj


def RBFkernel(vec1, vec2):
    return math.exp(-(np.linalg.norm(vec1 - vec2) ** 2) / (2 * sigmaKernel ** 2))

sigmaKernel = 0.4
pos = np.array([[0],[0]])
Traj = generateTrajectory(0,pos)

fig = plt.figure()
plt.plot(Traj[0,:],Traj[1,:])
plt.title("True field")
plt.show()