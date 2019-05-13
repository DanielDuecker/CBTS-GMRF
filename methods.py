import math
import matplotlib.pyplot as plt
import numpy as np

import parameters as par


def getMeasurement(xMeas, yMeas, trueField, noiseVariance):
    noise = np.random.normal(0, math.sqrt(noiseVariance))
    return (trueField.field(xMeas, yMeas) + noise)[0]


def mapConDis(gmrf, xMeas, yMeas):
    # Initialize j-th row of mapping matrix Phi
    Phi = np.zeros((1, gmrf.nP))
    Phi = np.hstack((Phi, np.zeros((1, gmrf.nBeta))))

    # Get grid position relative to surrounding vertices
    xRel = (xMeas - gmrf.xMin) % gmrf.dx - gmrf.dx / 2
    yRel = (yMeas - gmrf.yMin) % gmrf.dy - gmrf.dy / 2

    # Get index of upper left neighbor 
    xPos = int((xMeas - gmrf.xMin) / gmrf.dx)
    yPos = int((yMeas - gmrf.yMin) / gmrf.dy)

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
        if (i % gmrf.nX) == (gmrf.nX - 1):  # right border
            if i <= (gmrf.nP - gmrf.nX):
                Lambda[i, i + 1] = 0
    return Lambda


def getNextState(x, y, xBefore, yBefore, maxStepsize, gmrf):
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

    return (xNext, yNext)

def plotFields(fig, x, y, trueField, gmrf, controller, iterVec, timeVec, xHist, yHist):
    plt.clf()
    plt.ion()

    # Plotting ground truth
    ax1 = fig.add_subplot(221)
    ax1.contourf(x, y, trueField.field(x, y))
    plt.title("True field")

    # Plotting conditioned mean
    ax2 = fig.add_subplot(222)
    ax2.contourf(gmrf.x, gmrf.y, gmrf.meanCond[0:gmrf.nP].reshape(gmrf.nY, gmrf.nX))
    plt.xlabel("x in m")
    plt.ylabel("y in m")
    plt.title("Mean of belief")

    # Plotting covariance matrix
    ax3 = fig.add_subplot(223)
    ax3.contourf(gmrf.x, gmrf.y, gmrf.diagCovCond[0:gmrf.nP].reshape(gmrf.nY, gmrf.nX))
    ax3.plot(xHist, yHist, 'black')
    if par.PIControl:
        ax3.plot(controller.xTraj,controller.yTraj,'blue')
        for k in range(par.K):
            ax3.plot(controller.xPathRollOut[:, k], controller.yPathRollOut[:, k], 'grey')
        #TODO: Delete
        #ax3.plot(controller.xPathRollOut[:, 0], controller.yPathRollOut[:, 0], 'red')
        #ax3.plot(controller.xPathRollOut[:, 1], controller.yPathRollOut[:, 1], 'orange')
        #ax3.plot(controller.xPathRollOut[:, 2], controller.yPathRollOut[:, 2], 'green')
    plt.xlabel("x in m")
    plt.ylabel("y in m")
    plt.title("Uncertainty belief")

    # Plotting time consumption
    ax4 = fig.add_subplot(224)
    ax4.plot(iterVec, timeVec, 'black')
    plt.xlabel("Iteration index")
    plt.ylabel("calculation time in s")
    plt.title("Update calculation time over iteration index")

    fig.canvas.draw()

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
