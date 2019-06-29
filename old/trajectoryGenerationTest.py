import numpy as np
import math
import matplotlib.pyplot as plt

def generateTrajectory(alpha,posX,posY,derivX,derivY,theta,nTrajPoints):
    # theta = [ax ay bx by cx]
    # beta =    [dx cx bx ax]
    #           [dy cy by ay]
    # dx = posX, dy = posY, cx = dC1/du|u=0 = derivX, cy = dC2/du|u=0 = derivY
    ax = theta[0, 0]
    ay = theta[0, 1]
    bx = theta[0, 2]
    by = theta[0, 3]
    cx = derivX
    cy = derivY
    dx = posX
    dy = posY
    Nextcx = 3*ax+2*bx+cx
    Nextcy = 3*ay+2*by+cy
    beta = np.array([[dx, cx, bx, ax], [dy, cy, by, ay]])
    print("used alpha:",alpha*180/math.pi)
    tau = np.zeros((2, nTrajPoints))
    for i in range(nTrajPoints):
        u = i / (nTrajPoints - 1)
        tau[:, i] = np.dot(beta, np.array([[1], [u], [u ** 2], [u ** 3]]))[:, 0]

    alphaEnd = math.atan2((3 * ay + 2 * by + cy),(3 * ax + 2 * bx + cx))

    print("alphaEnd_before_check:",alphaEnd*180/math.pi)
    # Check if correct alpa is returned (otherwise add pi to it)
    #segmentOrientation = math.atan((tau[0,-2]-tau[0,-1])/(tau[1,-1]-tau[1,-2]))
    #print("segmentOrientation:",segmentOrientation*180/math.pi)
    #if not segmentOrientation <= alphaEnd <= segmentOrientation+math.pi:
     #   alphaEnd += math.pi
    #print("alphaEnd_after_check:",alphaEnd*180/math.pi)

    return tau, alphaEnd, Nextcx, Nextcy

#bx, by, cx
#theta = np.array([[1,1,1]])
thetaRange = np.linspace(-1,1,100)

plt.show()
for i in range(100):
    x = 5
    y = 5
    alpha = -math.pi / 4
    Nextcx = -1
    Nextcy = -1
    for j in range(1):
        #theta = np.array([[np.random.choice(thetaRange),np.random.choice(thetaRange),np.random.choice(thetaRange),np.random.choice(thetaRange)]])
        test = np.random.choice(thetaRange)
        #test = 1/2
        if test < 0:
            bx = np.sign(Nextcx)*test
            by = 0
        elif test >= 0:
            bx = 0
            by = -np.sign(Nextcy)*test
        theta = np.array([[0,0,bx,by]])
        print(theta)
        tau,alphaEnd,Nextcx,Nextcy = generateTrajectory(alpha,x,y,Nextcx,Nextcy,theta,10)
        plt.plot(tau[0,:],tau[1,:])
        x = tau[0,-1]
        y = tau[1,-1]
        alpha = alphaEnd
plt.axis('square')
plt.show()

