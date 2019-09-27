import matplotlib.pyplot as plt
import pickle

#Direct input
plt.rcParams['text.latex.preamble']=[r"\usepackage{lmodern}"]
#Options
params = {'text.usetex' : True,
          'font.size' : 12,
          'font.family' : 'lmodern',
          }
plt.rcParams.update(params)

plt.figure(1,figsize=(10,5))

saveToFile = True
index = 2

#fileNameRaw = input('Enter objs_other[..].pkl file: ')
#fileNameObject = fileNameRaw.split("'")[1]
fileNameObject = '1.pkl'
with open(fileNameObject,'rb') as f:
    xR, yR, trueFieldR, controllerR, CBTSR, timeVecR, xHistR, yHistR,  wrmseDictR, rmseDictR, wTotalVarDictR, totalVarDictR = pickle.load(f)

#fileNameRaw = input('Enter obs_i_gmrf[..].pkl file: ')
#fileNameGMRF = fileNameRaw.split("'")[1]
fileNameGMRF = '2.pkl'
with open(fileNameGMRF,'rb') as f:
    gmrf = pickle.load(f)

#simName = fileNameObject.split('other_')[1].split('.pkl')[0]
simName = 'test'
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

def plotBelief():
    CS = plt.contourf(gmrf.x[gmrf.nEdge:-gmrf.nEdge], gmrf.y[gmrf.nEdge:-gmrf.nEdge],
                      gmrf.meanCond[0:gmrf.nP].reshape(gmrf.nY, gmrf.nX)[gmrf.nEdge:-gmrf.nEdge,
                      gmrf.nEdge:-gmrf.nEdge],
                      levels=trueField.fieldLevels)
    for a in CS.collections:
        a.set_edgecolor('face')
    plt.plot(xHist, yHist, 'black')
    plt.xlabel("x in m")
    plt.ylabel("y in m")

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

plotBelief()
plt.savefig('plot/test.pdf',dpi=1000,bbox_inches='tight')
#tikzplotlib.save(simName + '.tikz')
