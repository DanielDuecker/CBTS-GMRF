# First 2D-GMRF Implementation

def main(par):
    #import cProfile
    #pr = cProfile.Profile()
    #pr.enable()

    import time

    import matplotlib.pyplot as plt
    import numpy as np
    import control
    import functions
    from classes import agent
    from classes import gmrf
    from classes import stkf
    from classes import trueField
    import math

    """Agent"""
    auv = agent(par, par.x0, par.y0, par.alpha0)
    xHist = [par.x0]  # x-state history vector
    yHist = [par.y0]  # y-state history vector

    """Plotting grid"""
    x = np.arange(par.xMin, par.xMax, par.dX)
    y = np.arange(par.yMin, par.yMax, par.dY)

    """Time"""
    timeVec = []

    """Performance measurement"""
    diffMeanVec = []
    totalVarVec = []

    """GMRF representation"""
    gmrf1 = gmrf(par,par.nGridX,par.nGridY)

    """PI2 Controller"""
    controller = control.piControl(par,)

    """Ground Truth"""
    trueField = trueField(par,par.fieldType)

    """STKF extension of gmrf"""
    stkf1 = stkf(par,gmrf1)

    """"Continuous Belief Tree Search"""
    CBTS1 = control.CBTS(par,)
    bestTraj = np.zeros((2, 1))
    trajIndex = 1 # current position in trajectory vector (0 is initial position)

    """Initialize plot"""
    if par.plot:
        fig0 = plt.figure(0)
        plt.clf()
        plt.ion()
        # plt.axis('equal')
        functions.plotFields(par, fig0, x, y, trueField, gmrf1, controller, CBTS1, timeVec, xHist, yHist)
        fig0.canvas.draw()
        plt.show()

    """Get first measurement:"""
    xMeas = par.x0
    yMeas = par.y0
    zMeas = np.zeros((par.nMeas, 1))  # Initialize measurement vector and mapping matrix
    Phi = np.zeros((par.nMeas, gmrf1.nP + gmrf1.nBeta))
    zMeas[0] = functions.getMeasurement(xMeas, yMeas, trueField, par.ov2Real)
    Phi[0, :] = functions.mapConDis(gmrf1, xMeas, yMeas)
    """Update and plot field belief"""
    for i in range(par.nIter - 1):
        print("Iteration ", i+1, " of ", par.nIter, ".")
        t = i * par.dt

        timeBefore = time.time()

        """Update belief"""
        if par.belief == 'stkf':
            stkf1.kalmanFilter(t, xMeas, yMeas, zMeas[i])
        elif par.belief == 'seqBayes':
            gmrf1.seqBayesianUpdate(zMeas[i], Phi[i, :])
        elif par.belief == 'regBayes' or par.belief == 'regBayesTrunc':
            gmrf1.bayesianUpdate(zMeas[0:(i+1)], Phi[0:(i+1), :])
        else:
            return("Error! No update method selected")

        """Controller"""
        if par.control == 'pi2':
            # Get next state according to PI Controller
            xMeas, yMeas = controller.getNewState(auv, gmrf1)
        elif par.control == 'cbts':
            if i % par.nMeasPoints == 0:
                bestTraj, auv.derivX, auv.derivY = CBTS1.getNewTraj(auv, gmrf1)
                trajIndex = 1
            auv.x = bestTraj[0, trajIndex]
            auv.y = bestTraj[1, trajIndex]
            xMeas = auv.x
            yMeas = auv.y
            trajIndex += 1
        elif par.control == 'randomWalk':
            # Get next measurement according to dynamics, stack under measurement vector
            xMeas, yMeas = functions.randomWalk(par, xMeas, yMeas, gmrf1)
        else:
            return("Error! No controller selected")

        xHist.append(xMeas)
        yHist.append(yMeas)
        zMeas[(i + 1) % par.nMeas] = functions.getMeasurement(xMeas, yMeas, trueField, par.ov2Real)

        """Map measurement to surrounding grid vertices and stack under Phi matrix"""
        Phi[(i + 1) % par.nMeas, :] = functions.mapConDis(gmrf1, xMeas, yMeas)

        """If truncated measurements are used, set conditioned mean and covariance as prior"""
        if par.belief == 'regBayesTrunc':
            if (i + 1) % par.nMeas == 0:
                gmrf1.covPrior = gmrf1.covCond
                gmrf1.meanPrior = gmrf1.meanCond

        """Time measurement"""
        timeAfter = time.time()
        timeVec.append(timeAfter - timeBefore)

        """Measure performance"""
        diffMean, totalVar = functions.measurePerformance(gmrf1, trueField)
        diffMeanVec.append(diffMean)
        totalVarVec.append(totalVar)
        if par.plotOptions.showPerformance:
            fig3 = plt.figure(3)
            plt.clf()
            plt.show()
            functions.plotPerformance(diffMeanVec, totalVarVec)
            fig3.canvas.draw()


        """Plotting"""
        if par.plot:
            plt.figure(0)
            plt.clf()
            functions.plotFields(par,fig0, x, y, trueField, gmrf1, controller, CBTS1, timeVec, xHist, yHist)
            fig0.canvas.draw()

        """Update ground truth:"""
        if par.temporal:
            trueField.updateField(par, i)

    #pr.disable()
    #pr.print_stats(sort='cumtime')

    if par.plot:
        plt.show(block=True)

    return x, y, trueField, gmrf1, controller, CBTS1, timeVec, xHist, yHist, diffMeanVec, totalVarVec
#functions.plotFields(fig, x, y, trueField, gmrf1, controller, CBTS1, iterVec, timeVec, xHist, yHist)
#plt.show(block=True)

# TODO Find reason for curved performance
# TODO use stkf in controller update
# TODO Fix distance between measurements on trajectory
# TODO Learning circular field
# TODO Try Car(2) precision matrix
# TODO use of sparse commands
# TODO Check computation time -> python library?
# TODO -> Change implementation of belief update at each node
# TODO tidy up code, consistent classes and paramater policy
# TODO Check reason for vertices in PI2
# TODO Compare to PI2
# TODO Use generic gmrf implementation (also for action reward mapping)
# TODO maybe use cubic splines or kernel trajs

# DONE
# TODO Check again feasability of trajs
# TODO add outer grid
# TODO boundary conditions
# TODO Use function for rotating field
# TODO Check first mean beliefs in STKF
# TODO maybe discount future rewards for exploration
# TODO Check visit counter
# TODO Improve negative rewards
# TODO Check if theta are dynamically feasible -> Done by only using specific input combinations which are incorporated
#       in a lower dimensional input
# TODO Try small thetas for less branching and deeper
# TODO Use current belief mean in reward function -> more exploitation
# TODO Check mean field for peak Value -> due to noise
# TODO Show plot of acquisiton function
