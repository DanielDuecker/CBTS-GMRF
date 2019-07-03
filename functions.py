import math
import matplotlib.pyplot as plt
import numpy as np

def getMeasurement(xMeas, yMeas, trueField, noiseVariance):
    noise = np.random.normal(0, math.sqrt(noiseVariance))
    return (trueField.getField(xMeas, yMeas) + noise)


def mapConDis(gmrf, xMeas, yMeas):
    # Initialize j-th row of mapping matrix Phi
    Phi = np.zeros((1, gmrf.nP))
    Phi = np.hstack((Phi, np.zeros((1, gmrf.nBeta)))) #TODO Change this

    # Get grid position relative to surrounding vertices
    xRel = (xMeas - gmrf.xMinEdge) % gmrf.dx - gmrf.dx / 2
    yRel = (yMeas - gmrf.yMinEdge) % gmrf.dy - gmrf.dy / 2

    # Get index of upper left neighbor
    xPos = int((xMeas - gmrf.xMinEdge) / gmrf.dx)
    yPos = int((yMeas - gmrf.yMinEdge) / gmrf.dy)

    # Local coordinate system is different from Geist! (e_y=-e_y_Geist)
    # because now mean vector is [vertice0,vertice1,vertice3,...])
    # Calculate weights at neighbouring positions
    try:
        Phi[0, (yPos + 1) * gmrf.nX + xPos] = 1 / (gmrf.dx * gmrf.dy) * (xRel - gmrf.dx / 2) * (
            -yRel - gmrf.dy / 2)  # lower left
        Phi[0, (yPos + 1) * gmrf.nX + xPos + 1] = -1 / (gmrf.dx * gmrf.dy) * (xRel + gmrf.dx / 2) * (
            -yRel - gmrf.dy / 2)  # lower right
        Phi[0, yPos * gmrf.nX + xPos + 1] = 1 / (gmrf.dx * gmrf.dy) * (xRel + gmrf.dx / 2) * (
            -yRel + gmrf.dy / 2)  # upper right
        Phi[0, yPos * gmrf.nX + xPos] = -1 / (gmrf.dx * gmrf.dy) * (xRel - gmrf.dx / 2) * (
            -yRel + gmrf.dy / 2)  # upper left

    except:
        print("Error! Agent is out of bound with state (",xMeas,",",yMeas,")")
    return Phi


def getPrecisionMatrix(gmrf):
    diagonalValue = 4  # needs to be high enough in order to create a strictly diagonal dominant matrix
    Lambda = diagonalValue * np.eye(gmrf.nP) - np.eye(gmrf.nP, k=gmrf.nX) - np.eye(gmrf.nP, k=-gmrf.nX)
    Lambda -= np.eye(gmrf.nP, k=1) + np.eye(gmrf.nP, k=-1)

    # set precision entry to zero if left or right border is reached
    # (since there is no connection between the two edge vertices)
    for i in range(gmrf.nP):
        if (i % gmrf.nX) == 0:  # left border
            if i >= gmrf.nX:
                Lambda[i, i - 1] = 0
                Lambda[i, i - 1] = 0
        if (i % gmrf.nX) == (gmrf.nX - 1):  # right border
            if i <= (gmrf.nP - gmrf.nX):
                Lambda[i, i + 1] = 0
    return Lambda


def getNextState(par, x, y, xBefore, yBefore, maxStepsize, gmrf):
    xNext = xBefore
    yNext = yBefore

    stepsize = maxStepsize * np.random.rand()

    # avoid going back to former location:
    while xNext == xBefore and yNext == yBefore:
        xNext = np.random.choice([x - stepsize, x + stepsize])
        yNext = np.random.choice([y - stepsize, y + stepsize])

    # x and y are switched because of matrix/plot relation
    (yUncertain, xUncertain) = np.unravel_index(
        np.argmax(gmrf.diagCovCond[0:gmrf.nP].reshape(gmrf.nY, gmrf.nX), axis=None), (gmrf.nY, gmrf.nX))

    dist = np.sqrt((xUncertain - x) ** 2 + (yUncertain - y) ** 2)

    if np.random.rand() < par.exploitingRate:
        xNext = x + par.xVel * (xUncertain - x) / dist
        yNext = y + par.yVel * (yUncertain - y) / dist
        if dist == 0:
            xNext = x
            yNext = y

    if xNext < gmrf.xMin:
        xNext = x + par.xVel
    elif xNext > gmrf.xMax:
        xNext = x - par.xVel

    if yNext < gmrf.yMin:
        yNext = y + par.yVel
    elif yNext > gmrf.yMax:
        yNext = y - par.yVel

    return xNext, yNext


def plotFields(par, fig, x, y, trueField, gmrf, controller,CBTS1, timeVec, xHist, yHist):
    # Plotting ground truth
    ax1 = fig.add_subplot(221)
    Z = trueField.getField(x,y)
    CS = ax1.contourf(x, y, Z, levels=trueField.fieldLevels)
    for a in CS.collections:
        a.set_edgecolor('face')
    ax1.plot(xHist, yHist, 'black')
    plt.title("True Field")

    # Plotting conditioned mean
    ax2 = fig.add_subplot(222)
    CS = ax2.contourf(gmrf.x[gmrf.nEdge:-gmrf.nEdge], gmrf.y[gmrf.nEdge:-gmrf.nEdge],
                 gmrf.meanCond[0:gmrf.nP].reshape(gmrf.nY, gmrf.nX)[gmrf.nEdge:-gmrf.nEdge,gmrf.nEdge:-gmrf.nEdge],
                 levels=trueField.fieldLevels)
    for a in CS.collections:
        a.set_edgecolor('face')
    ax2.plot(xHist, yHist, 'black')
    plt.xlabel("x in m")
    plt.ylabel("y in m")
    plt.title("Mean of Belief")

    # Plotting covariance matrix
    ax3 = fig.add_subplot(223)
    CS = ax3.contourf(gmrf.x[gmrf.nEdge:-gmrf.nEdge], gmrf.y[gmrf.nEdge:-gmrf.nEdge],
                 gmrf.diagCovCond[0:gmrf.nP].reshape(gmrf.nY, gmrf.nX)[gmrf.nEdge:-gmrf.nEdge,gmrf.nEdge:-gmrf.nEdge],
                 levels=gmrf.covLevels)
    for a in CS.collections:
        a.set_edgecolor('face')
    if par.control == 'pi2':
        ax3.plot(controller.xTraj,controller.yTraj,'blue')
        for k in range(par.K):
            ax3.plot(controller.xPathRollOut[:, k], controller.yPathRollOut[:, k], 'grey')

    elif par.control == 'cbts':
        for k in range(CBTS1.xTraj.shape[1]-1):
            ax3.plot(CBTS1.xTraj[:, k+1], CBTS1.yTraj[:, k+1],'grey')
    ax3.plot(xHist, yHist, 'black')

    plt.xlabel("x in m")
    plt.ylabel("y in m")
    plt.title("Uncertainty Belief")

    # Plotting time consumption
    ax4 = fig.add_subplot(224)
    plt.cla()
    ax4.plot(timeVec)
    plt.xlabel("Iteration Index")
    plt.ylabel("Time in s")
    plt.title("Computation Time")

def plotPolicy(par, GP,thetaPredict,mu):
    fig = plt.figure(1)
    plt.clf()
    plt.show()

    if par.trajOrder == 2:
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(GP.trainInput[:, 0], GP.trainInput[:, 1], GP.trainOutput[:, 0], "g.")
        ax.plot(thetaPredict[:, 0], thetaPredict[:, 1], mu[:, 0], "r.")
        ax.set_xlabel("Theta[0]")
        ax.set_ylabel("Theta[1]")
        ax.set_zlabel("Reward")
    elif par.trajOrder == 1:
        ax = fig.add_subplot(111)
        ax.plot(GP.trainInput[:, 0], GP.trainOutput[:, 0], "g.")
        ax.plot(thetaPredict[:, 0], mu[:, 0], "r.")
    fig.canvas.draw()

def plotRewardFunction(par,gmrf):
    fig2 = plt.figure(2)
    plt.clf()
    plt.show()
    plt.title("Current Rewards for Each Position")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()

    X,Y = np.meshgrid(gmrf.x[gmrf.nEdge:-gmrf.nEdge],gmrf.y[gmrf.nEdge:-gmrf.nEdge])
    r = gmrf.diagCovCond[0:gmrf.nP].reshape(gmrf.nY, gmrf.nX)[gmrf.nEdge:-gmrf.nEdge,gmrf.nEdge:-gmrf.nEdge] \
        + par.UCBRewardFactor * gmrf.meanCond[0:gmrf.nP].reshape(gmrf.nY, gmrf.nX)[gmrf.nEdge:-gmrf.nEdge,gmrf.nEdge:-gmrf.nEdge]
    ax = fig2.add_subplot(111)
    ax.contourf(X,Y,r)
    fig2.canvas.draw()

def sanityCheck(xVec,yVec,gmrf):
    for x in xVec:
        if x < gmrf.xMin:
            return False
        elif x > gmrf.xMax:
            return False

    for y in yVec:
        if y < gmrf.yMin:
            return False
        elif y > gmrf.yMax:
            return False
    return True

def measurePerformance(gmrf,trueField):
    diffMean = np.sum(trueField.getField(gmrf.x[gmrf.nEdge:-gmrf.nEdge],gmrf.y[gmrf.nEdge:-gmrf.nEdge])
                      -gmrf.meanCond[0:gmrf.nP].reshape(gmrf.nY, gmrf.nX)[gmrf.nEdge:-gmrf.nEdge,gmrf.nEdge:-gmrf.nEdge])
    totalVar = np.sum(gmrf.covCond)
    return diffMean,totalVar

def plotPerformance(diffMean,totalVar):
    plt.title('Performance Measurement')
    plt.subplot(211)
    plt.plot(diffMean)
    plt.xlabel('Iteration Index')
    plt.ylabel('Difference Between Ground Truth and Belief')
    plt.subplot(212)
    plt.plot(totalVar)
    plt.xlabel('Iteration Index')
    plt.ylabel('Total Belief Uncertainty')