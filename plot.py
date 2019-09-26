import matplotlib.pyplot as plt
import pickle
import matplotlib2tikz

saveToFile = True

fileNameObject = input('Enter object file: ')
with open(fileNameObject,'rb') as f:
    xR, yR, trueFieldR, controllerR, CBTSR, timeVecR, xHistR, yHistR,  wrmseDictR, rmseDictR, wTotalVarDictR, totalVarDictR = pickle.load(f)

fileNameGMRF = input('Enter gmrf file: ')
index = 3
with open(fileNameGMRF,'rb') as f:
    gmrf = pickle.load(f)

x = xR[index]
y = yR[index]
trueField = trueFieldR[index]
controller = controllerR[index]
CBTS = CBTSR[index]
timeVec = timeVecR[index]
xHist = xHistR[index]
yHist = yHistR[index]
wrmseDict = wrmseDictR[index]
rmseDict = rmseDictR[index]
wTotalVarDict = wTotalVarDictR[index]
totalVarDict = totalVarDictR[index]

par = CBTS.par

def plotAllFields():
    fig = plt.figure(0)
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
        for k in range(CBTS.xTraj.shape[1] - 1):
            ax3.plot(CBTS.xTraj[:, k + 1], CBTS.yTraj[:, k + 1], 'grey')
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

def plotBelief():
    fig = plt.figure()
    CS = plt.contourf(gmrf.x[gmrf.nEdge:-gmrf.nEdge], gmrf.y[gmrf.nEdge:-gmrf.nEdge],
                      gmrf.meanCond[0:gmrf.nP].reshape(gmrf.nY, gmrf.nX)[gmrf.nEdge:-gmrf.nEdge,
                      gmrf.nEdge:-gmrf.nEdge],
                      levels=trueField.fieldLevels)
    for a in CS.collections:
        a.set_edgecolor('face')
    plt.plot(xHist, yHist, 'black')
    plt.xlabel("x in m")
    plt.ylabel("y in m")
    plt.title("Mean of Belief")

plotBelief()

matplotlib2tikz.save("testNEW.tikz")
