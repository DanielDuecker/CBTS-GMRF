import math

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from scipy import stats
import copy

import classes


def getMeasurement(xMeas, yMeas, trueField, noiseVariance):
    noise = np.random.normal(0, math.sqrt(noiseVariance))
    return trueField.getField(xMeas, yMeas) + noise


def mapConDis(gmrf, xMeas, yMeas, borderEnforcement = True):
    # Initialize j-th row of mapping matrix Phi
    Phi = np.zeros((1, gmrf.nP))
    Phi = np.hstack((Phi, np.zeros((1, gmrf.nBeta))))  # TODO Change this

    if borderEnforcement:
        if xMeas < gmrf.xMin:
            return Phi
        elif xMeas > gmrf.xMax:
            return Phi
        if yMeas < gmrf.yMin:
            return Phi
        elif yMeas > gmrf.yMax:
            return Phi

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
        print("Error! Agent is out of bound with state (", xMeas, ",", yMeas, ")")
        Phi = np.zeros((1, gmrf.nP))
        Phi = np.hstack((Phi, np.zeros((1, gmrf.nBeta))))
    return Phi


def getPrecisionMatrix(gmrf):
    kappa = 1e-4#0.1  # needs to be high enough in order to create a strictly diagonal dominant matrix
    Lambda = (4 + kappa**2) * np.eye(gmrf.nP) - np.eye(gmrf.nP, k=gmrf.nX) - np.eye(gmrf.nP, k=-gmrf.nX)
    Lambda -= np.eye(gmrf.nP, k=1) + np.eye(gmrf.nP, k=-1)

    for i in range(gmrf.nP):

        # set precision value on diagonal to 3 if index at lower and upper border
        if i < gmrf.nX:
            Lambda[i, i] = 3 + kappa**2
        elif i >= (gmrf.nP - gmrf.nX):
            Lambda[i, i] = 3 + kappa**2

        # if index switched from left to right border, set precision value on side diagonal to zero since ther
        # is no physical connection. Also set precision value at diagonal to 3 if a left or right border is reached
        if (i % gmrf.nX) == 0:  # left border
            if i >= gmrf.nX:
                Lambda[i, i - 1] = 0
                Lambda[i, i] = 3 + kappa**2
        if (i % gmrf.nX) == (gmrf.nX - 1):  # right border
            if i <= (gmrf.nP - gmrf.nX):
                Lambda[i, i + 1] = 0
                Lambda[i, i] = 3 + kappa**2

        # Set diagonal corner precision value to 2
        Lambda[0, 0] = 2 + kappa**2
        Lambda[-1, -1] = 2 + kappa**2
        Lambda[gmrf.nX - 1, gmrf.nX - 1] = 2 + kappa**2
        Lambda[-gmrf.nX, -gmrf.nX] = 2 + kappa**2

    return sp.csr_matrix(Lambda)


def randomWalk(par, auv, gmrf):
    alphaNext = auv.alpha + np.random.normal(0, par.noiseRandomWalk)
    xNext = auv.x + par.maxStepsize * math.cos(alphaNext)
    yNext = auv.y + par.maxStepsize * math.sin(alphaNext)

    if xNext < gmrf.xMin:
        xNext = gmrf.xMin
        alphaNext += math.pi
    elif xNext > gmrf.xMax:
        xNext = gmrf.xMax
        alphaNext += math.pi

    if yNext < gmrf.yMin:
        yNext = gmrf.yMin
        alphaNext += math.pi
    elif yNext > gmrf.yMax:
        yNext = gmrf.yMax
        alphaNext += math.pi

    return xNext, yNext, alphaNext


def plotFields(par, fig, x, y, trueField, gmrf, controller, CBTS1, timeVec, xHist, yHist):
    # Plotting ground truth
    ax1 = fig.add_subplot(221)
    Z = trueField.getField(x, y)
    CS = ax1.contourf(x, y, Z, levels=trueField.fieldLevels)
    for a in CS.collections:
        a.set_edgecolor('face')

    ax1.plot(xHist, yHist, 'black')
    plt.title("True Field")

    # Plotting conditioned mean
    ax2 = fig.add_subplot(222)
    CS = ax2.contourf(gmrf.x[gmrf.nEdge:-gmrf.nEdge], gmrf.y[gmrf.nEdge:-gmrf.nEdge],
                      gmrf.meanCond[0:gmrf.nP].reshape(gmrf.nY, gmrf.nX)[gmrf.nEdge:-gmrf.nEdge,
                      gmrf.nEdge:-gmrf.nEdge],
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
                      gmrf.diagCovCond[0:gmrf.nP].reshape(gmrf.nY, gmrf.nX)[gmrf.nEdge:-gmrf.nEdge,
                      gmrf.nEdge:-gmrf.nEdge],
                      levels=gmrf.covLevels)
    for a in CS.collections:
        a.set_edgecolor('face')
    if par.control == 'pi2':
        ax3.plot(controller.xTraj, controller.yTraj, 'blue')
        for k in range(par.K):
            ax3.plot(controller.xPathRollOut[:, k], controller.yPathRollOut[:, k], 'grey')
    elif par.control == 'cbts':
        for k in range(CBTS1.xTraj.shape[1] - 1):
            ax3.plot(CBTS1.xTraj[:, k + 1], CBTS1.yTraj[:, k + 1], 'grey')
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


def plotPolicy(par, GP, thetaPredict, mu, var):
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
        plt.scatter(GP.trainInput[:, 0], GP.trainOutput[:, 0], color='black', s=10, marker='x', zorder=3,
                    label='Training Input')
        plt.scatter(thetaPredict[:, 0], mu[:, 0], color='red', s=10, label='Test Input')
        h = mu[:, 0] + par.kappa * var.diagonal().reshape(par.nThetaSamples)
        plt.scatter(thetaPredict[:, 0], h, color='blue', s=10, label='Acquisition Function Samples')
        plt.xlabel('Selected Action')
        plt.ylabel('Reward')
    fig.canvas.draw()


def plotRewardFunction(par, gmrf):
    fig2 = plt.figure(2)
    plt.clf()
    plt.show()
    plt.title("Current Rewards for Each Position")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()

    X, Y = np.meshgrid(gmrf.x[gmrf.nEdge:-gmrf.nEdge], gmrf.y[gmrf.nEdge:-gmrf.nEdge])
    r = gmrf.diagCovCond[0:gmrf.nP].reshape(gmrf.nY, gmrf.nX)[gmrf.nEdge:-gmrf.nEdge, gmrf.nEdge:-gmrf.nEdge] \
        + par.UCBRewardFactor * gmrf.meanCond[0:gmrf.nP].reshape(gmrf.nY, gmrf.nX)[gmrf.nEdge:-gmrf.nEdge,
                                gmrf.nEdge:-gmrf.nEdge]
    ax = fig2.add_subplot(111)
    ax.contourf(X, Y, r)
    fig2.canvas.draw()


def sanityCheck(xVec, yVec, gmrf):
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


def obstacleCheck(xVec, yVec, gmrf):
    if gmrf.par.obstacle:
        for x in xVec:
            for y in yVec:
                if gmrf.par.xGateLeft <= x <= gmrf.par.xGateRight:
                    if y <= gmrf.par.yGateLower or y >= gmrf.par.yGateUpper:
                        return True
    return False

def measurePerformance(gmrf,trueField):
    copyDiagCovCond = copy.copy(gmrf.diagCovCond[0:gmrf.nP])
    true = copy.copy(trueField.getField(gmrf.x[gmrf.nEdge:-gmrf.nEdge], gmrf.y[gmrf.nEdge:-gmrf.nEdge]))
    belief = copy.copy(gmrf.meanCond[0:gmrf.nP].reshape(gmrf.nY, gmrf.nX)[gmrf.nEdge:-gmrf.nEdge, gmrf.nEdge:-gmrf.nEdge])
    weights = (true-np.min(true)*np.ones(true.shape))/(np.max(true)-np.min(true))

    # set predictive variance to zero which belongs to restricted area
    if gmrf.par.obstacle:
        for ix in range(len(gmrf.x[gmrf.nEdge:-gmrf.nEdge])):
            for iy in range(len(gmrf.y[gmrf.nEdge:-gmrf.nEdge])):
                x = gmrf.x[gmrf.nEdge:-gmrf.nEdge][ix]
                y = gmrf.y[gmrf.nEdge:-gmrf.nEdge][iy]
                if obstacleCheck([x],[y],gmrf):
                    index = np.argmax(mapConDis(gmrf,x,y,False))
                    copyDiagCovCond[index] = 0

    wrmseMean = math.sqrt(np.mean(np.multiply((belief-true)**2, weights)))
    wTotalVar = np.sum(abs(np.multiply(copyDiagCovCond.reshape(gmrf.nY, gmrf.nX)[gmrf.nEdge:-gmrf.nEdge, gmrf.nEdge:-gmrf.nEdge], weights)))

    rmseMean = math.sqrt(np.mean((belief-true)**2))
    totalVar = np.sum(abs(copyDiagCovCond.reshape(gmrf.nY, gmrf.nX)[gmrf.nEdge:-gmrf.nEdge, gmrf.nEdge:-gmrf.nEdge]))
    return wrmseMean, rmseMean, wTotalVar, totalVar

def plotPerformance(rmseMean,totalVar):
    plt.title('Performance Measurement')
    plt.subplot(211)
    plt.plot(rmseMean)
    plt.xlabel('Iteration Index')
    plt.ylabel('Root Mean Squared Error Between Ground Truth and Belief')
    plt.subplot(212)
    plt.plot(totalVar)
    plt.xlabel('Iteration Index')
    plt.ylabel('Total Belief Uncertainty')

def plotWeightedPerformance(wrmseMean,wtotalVar):
    plt.title('Weighted Performance Measurement')
    plt.subplot(211)
    plt.plot(wrmseMean)
    plt.xlabel('Iteration Index')
    plt.ylabel('Weighted Root Mean Squared Error Between Ground Truth and Belief')
    plt.subplot(212)
    plt.plot(wtotalVar)
    plt.xlabel('Iteration Index')
    plt.ylabel('Total Belief Uncertainty')

def plotOverallPerformance(nIter, simCaseList, rmseDict, totalVarDict, weighted, type):
    x = np.linspace(0, nIter, nIter)
    plt.subplot(211)
    for sim in simCaseList:
        if type == 'median':
            rmse = np.median(rmseDict[sim], axis=0)
            lower = stats.iqr(rmseDict[sim], axis=0, rng=(25, 50))
            upper = stats.iqr(rmseDict[sim], axis=0, rng=(50, 75))
        elif type == 'mean':
            rmse = np.mean(rmseDict[sim], axis=0)
            lower = np.std(rmseDict[sim], axis=0)
            upper = lower
        plt.plot(x, rmse, label=sim)
        #plt.plot(x, rmse - lower, 'gray')
        #plt.plot(x, rmse + upper, 'gray')
        plt.fill_between(x,  rmse - lower, rmse + upper, cmap='twilight', alpha=0.4)
    plt.xlabel('Iteration Index')
    if weighted:
        plt.ylabel('Weighted RMSE')
    else:
        plt.ylabel('RMSE')
    plt.legend()

    y = np.linspace(0, nIter - 1, nIter - 1)
    plt.subplot(212)
    for sim in simCaseList:
        if type == 'median':
            var = np.median(totalVarDict[sim], axis=0)
            lower = stats.iqr(totalVarDict[sim], axis=0, rng=(25, 50))
            upper = stats.iqr(totalVarDict[sim], axis=0, rng=(50, 75))
        elif type == 'mean':
            var = np.mean(totalVarDict[sim], axis=0)
            lower = np.std(totalVarDict[sim], axis=0)
            upper = lower
        plt.plot(y, var, label=sim)
        #plt.plot(y, var - lower, 'gray')
        #plt.plot(y, var + upper, 'gray')
        plt.fill_between(y,  var - lower, var + upper, cmap='twilight', alpha=0.4)
    plt.xlabel('Iteration Index')
    if weighted:
        plt.ylabel('Weighted Predictive Variance')
    else:
        plt.ylabel('Predictive Variance')
    plt.legend()


def sampleGMRF(gmrf):
    """The sampled GMRFs without outer grids are used in the belief nodes in order to reduce the computation time"""
    newGMRF = classes.gmrf(gmrf.par, gmrf.par.nGridXSampled, gmrf.par.nGridYSampled, 0)
    gmrf.precCondSparseDense = np.array(gmrf.precCondSparse.todense())

    if gmrf.nBeta > 0:
        newGMRF.bSeq[-gmrf.nBeta:] = gmrf.bSeq[-gmrf.nBeta:]
        newGMRF.diagCovCond[-gmrf.nBeta:] = gmrf.diagCovCond[-gmrf.nBeta:]
        newGMRF.precCondSparse[-gmrf.nBeta:,-gmrf.nBeta:] = gmrf.precCondSparse[-gmrf.nBeta:,-gmrf.nBeta:]
    
    for i in range(newGMRF.nP):
        xIndex = i % newGMRF.nX
        yIndex = int(i / newGMRF.nX)
        Phi = mapConDis(gmrf, newGMRF.x[xIndex], newGMRF.y[yIndex])
        newGMRF.bSeq[i] = np.dot(Phi, gmrf.bSeq)
        newGMRF.precCondSparse[i, i] = np.dot(Phi, np.dot(gmrf.precCondSparseDense, Phi.T))
        newGMRF.diagCovCond[i] = np.dot(Phi, gmrf.diagCovCond)
        newGMRF.meanCond[i] = np.dot(Phi, gmrf.meanCond)
    newGMRF.diagCovCond = newGMRF.covCond.diagonal().reshape(newGMRF.nP + newGMRF.nBeta, 1)

    """Check sampling"""
    """
    fig = plt.figure(999)
    plt.clf()
    plt.ion()
    ax1 = fig.add_subplot(221)
    ax1.contourf(newGMRF.x, newGMRF.y,newGMRF.bSeq[0:newGMRF.nP].reshape(newGMRF.nY, newGMRF.nX))
    X,Y = np.meshgrid(newGMRF.x, newGMRF.y)
    ax1.scatter(Y,X,s=0.1)

    ax2 = fig.add_subplot(222)
    ax2.contourf(newGMRF.x, newGMRF.y,
                 newGMRF.covCond.diagonal()[0:newGMRF.nP].reshape(newGMRF.nY, newGMRF.nX))
    ax2.scatter(Y,X,s=0.1)

    ax3 = fig.add_subplot(223)
    if gmrf.nEdge == 0:
        ax3.contourf(gmrf.x, gmrf.y,gmrf.bSeq[0:gmrf.nP].reshape(gmrf.nY, gmrf.nX))
    else:
        ax3.contourf(gmrf.x[gmrf.nEdge:-gmrf.nEdge], gmrf.y[gmrf.nEdge:-gmrf.nEdge],gmrf.bSeq[0:gmrf.nP].reshape(gmrf.nY, gmrf.nX)[gmrf.nEdge:-gmrf.nEdge,gmrf.nEdge:-gmrf.nEdge])
    X,Y = np.meshgrid(gmrf.x[gmrf.nEdge:-gmrf.nEdge], gmrf.y[gmrf.nEdge:-gmrf.nEdge])
    ax3.scatter(Y,X,s=0.1)

    ax4 = fig.add_subplot(224)
    if gmrf.nEdge == 0:
        ax4.contourf(gmrf.x, gmrf.y,gmrf.diagCovCond[0:gmrf.nP].reshape(gmrf.nY, gmrf.nX))

    else:
        ax4.contourf(gmrf.x[gmrf.nEdge:-gmrf.nEdge], gmrf.y[gmrf.nEdge:-gmrf.nEdge],
                 gmrf.diagCovCond[0:gmrf.nP].reshape(gmrf.nY, gmrf.nX)[gmrf.nEdge:-gmrf.nEdge,gmrf.nEdge:-gmrf.nEdge])
    ax4.scatter(Y,X,s=0.1)

    fig.canvas.draw()
    plt.show()
    """

    return newGMRF
