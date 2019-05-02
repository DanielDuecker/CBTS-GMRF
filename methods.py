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
    Phi[0, (yPos + 1) * gmrf.nX + xPos] = 1 / (gmrf.dx * gmrf.dy) * (xRel - gmrf.dx / 2) * (
            -yRel - gmrf.dy / 2)  # lower left
    Phi[0, (yPos + 1) * gmrf.nX + xPos + 1] = -1 / (gmrf.dx * gmrf.dy) * (xRel + gmrf.dx / 2) * (
            -yRel - gmrf.dy / 2)  # lower right
    Phi[0, yPos * gmrf.nX + xPos + 1] = 1 / (gmrf.dx * gmrf.dy) * (xRel + gmrf.dx / 2) * (
            -yRel + gmrf.dy / 2)  # upper right
    Phi[0, yPos * gmrf.nX + xPos] = -1 / (gmrf.dx * gmrf.dy) * (xRel - gmrf.dx / 2) * (
            -yRel + gmrf.dy / 2)  # upper left

    return Phi


# def getPrecisionMatrix(gmrf):
#    diagQ = 2*np.eye(gmrf.nP)
#    Lambda = diagQ-np.eye(gmrf.nP,k=1)-np.eye(gmrf.nP,k=-1)
#    return Lambda

def getPrecisionMatrix(gmrf):
    #diagQ = 2 * np.eye(gmrf.nP)
    #Lambda = diagQ - np.eye(gmrf.nP, k=1) - np.eye(gmrf.nP, k=-1)
    Lambda = np.array([[5,-1,0,-1,0,0,0,0,0],
                    [-1,3,-1,0,-1,0,0,0,0],
                    [0,-1,2,0,0,-1,0,0,0],
                    [-1,0,0,3,-1,0,-1,0,0],
                    [0,-1,0,-1,4,-1,0,-1,0],
                    [0,0,-1,0,-1,3,0,0,-1],
                    [0,0,0,-1,0,0,2,-1,0],
                    [0,0,0,0,-1,0,-1,3,-1],
                    [0,0,0,0,0,-1,0,-1,2]])
    print(Lambda)
    #Lambda = np.zeros((gmrf.nY,gmrf.nX))
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
    (yUncertain,xUncertain) = np.unravel_index(np.argmax(gmrf.diagCovCond[0:gmrf.nP].reshape(gmrf.nY, gmrf.nX),axis=None),(gmrf.nY,gmrf.nX))

    dist = np.sqrt((xUncertain-x)**2+(yUncertain-y)**2)
    if np.random.rand() < par.exploitingRate:
        xNext = x + par.xVel*(xUncertain-x)/dist
        yNext = y + par.yVel*(yUncertain-y)/dist

    if xNext < gmrf.xMin:
        xNext = x + stepsize
    elif xNext > gmrf.xMax:
        xNext = x - stepsize

    if yNext < gmrf.yMin:
        yNext = y + stepsize
    elif yNext > gmrf.yMax:
        yNext = y - stepsize

    return xNext, yNext


def plotFields(fig, x, y, trueField, gmrf, iterVec, timeVec, xHist, yHist):
    plt.clf()
    plt.ion()

    # Plotting ground truth
    ax1 = fig.add_subplot(221)
    ax1.contourf(x, y, trueField.field(x,y))
    plt.title("True field")

    # Plotting conditioned mean
    ax2 = fig.add_subplot(222)
    ax2.contourf(gmrf.x, gmrf.y, gmrf.meanCond[0:gmrf.nP].reshape(gmrf.nY, gmrf.nX))
    plt.xlabel("x in m")
    plt.ylabel("y in m")
    plt.title("Mean of belief")

    # Plotting precision matrix
    ax3 = fig.add_subplot(223)
    ax3.contourf(gmrf.x, gmrf.y, gmrf.diagCovCond[0:gmrf.nP].reshape(gmrf.nY, gmrf.nX))
    ax3.plot(xHist, yHist, 'black')
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
