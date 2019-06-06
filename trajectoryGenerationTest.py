import numpy as np
import math
import matplotlib.pyplot as plt

def generateTrajectory(alpha,x,y, theta,nTrajPoints):
    # theta = [ax ay bx by cx]
    # beta =    [dx cx bx ax]
    #           [dy cy by ay]
    # dx = posX, dy = posY, cy/cx = tan(alpha)
    ax = theta[0, 4]
    ay = theta[0, 3]
    bx = theta[0, 0]
    by = theta[0, 1]
    cx = theta[0, 2]
    cy = cx * math.tan(alpha)
    print("init alpha in func:",math.atan(cy/cx)*180/math.pi)
    dx = x
    dy = y

    beta = np.array([[dx, cx, bx, ax], [dy, cy, by, ay]])

    alphaEnd = math.atan((3 * ay + 2 * by + cy) / (3 * ax + 2 * bx + cx))

    tau = np.zeros((2, nTrajPoints))
    for i in range(nTrajPoints):
        u = i / (nTrajPoints - 1)
        tau[:, i] = np.dot(beta, np.array([[1], [u], [u ** 2], [u ** 3]]))[:, 0]
    return tau, alphaEnd

#bx, by, cx
theta = np.array([[1,1,1]])
x = 0
y = 0
alpha = math.pi/4


for i in range(100):
    x = 0
    y = 0
    alpha = math.pi / 4
    for j in range(3):
        print("Initial alpha:",alpha*180/math.pi)
        theta = np.array([[np.random.rand(),np.random.rand(),np.random.rand(),0,0]])
        tau,alphaEnd = generateTrajectory(alpha,x,y,theta,500)
        plt.plot(tau[0,:],tau[1,:])
        x = tau[0,-1]
        y = tau[1,-1]
        alpha = alphaEnd
plt.axis('square')
plt.show()

