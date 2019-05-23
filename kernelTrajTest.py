import math
import numpy as np
import parameters as par
import matplotlib.pyplot as plt

def generateTrajectory(theta, pos):
    # create anchor points
    nAnchorPoints = 30
    trajectoryNoise = 0.05

    lMax = 10
    eps = 1
    delta = 1
    Traj = np.zeros((2, nAnchorPoints))

    anchorPoints = np.zeros((2, nAnchorPoints))
    anchorPoints[:, 0] = pos[:,0]
    y = np.zeros((1, nAnchorPoints))
    y[0, 0] = 0
    alpha = math.pi/4
    for i in range(nAnchorPoints - 1):
        alpha = alpha + 2*math.pi * np.random.normal(0,trajectoryNoise)
        anchorPoints[0, i + 1] = anchorPoints[0, i] + par.maxStepsize / nAnchorPoints * math.cos(alpha)
        anchorPoints[1, i + 1] = anchorPoints[1, i] + par.maxStepsize / nAnchorPoints * math.sin(alpha)
        y[0, i + 1] = y[0, i] + par.dt / nAnchorPoints
        # calculate weight vector
        GX = np.zeros((nAnchorPoints, nAnchorPoints))
        GY = np.zeros((nAnchorPoints, nAnchorPoints))

    mPi = np.zeros((nAnchorPoints, nAnchorPoints))
    for i in range(nAnchorPoints):
        for j in range(nAnchorPoints):
            GX[i, j] = RBFkernel(anchorPoints[:, i], anchorPoints[:, j])
            GY[i, j] = RBFkernel(y[0, i], y[0, j])
            mPi[i, j] = sum(np.random.choice(GX.flatten(), lMax)) / lMax

    Traj = np.zeros((2,nAnchorPoints))
    for iy in range(nAnchorPoints):
        Lambd = np.dot(np.linalg.inv(GX + eps * np.eye(nAnchorPoints)), mPi[:,iy])
        LambdGY = np.dot(Lambd.T, GY)
        factor1 = np.linalg.inv(np.dot(LambdGY, LambdGY.T) + eps)[0,0]
        factor2 = np.dot(LambdGY,Lambd[:,0])[0]
        wy = factor1*factor2*GY[:, iy].reshape(nAnchorPoints,1)
        Traj[:,iy] = sum(np.dot(anchorPoints, wy))
    return Traj


def RBFkernel(vec1, vec2):
    return math.exp(-(np.linalg.norm(vec1 - vec2) ** 2) / (2 * sigmaKernel ** 2))

sigmaKernel = 0.1
pos = np.array([[0],[0]])
Traj = generateTrajectory(0,pos)

fig = plt.figure()
plt.plot(Traj[0,:],Traj[1,:])
plt.title("True field")
plt.show()