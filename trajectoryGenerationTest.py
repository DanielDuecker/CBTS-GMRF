from parameters import par
from control import CBTS
from classes import node
from classes import agent
from classes import gmrf
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import math

matplotlib.use('TkAgg')

N = 3
depth = 3
par = par('seqBayes', 'cbts', 'noUpdates', 'random', False, False, 1000, 0, 0, 0, 0, 0, 6, 3, 50, 50, 0.05, 0.5, 0.9)
cbts = CBTS(par)
auv = agent(par, par.x0, par.y0, par.alpha0)
gmrf1 = gmrf(par, par.nGridX, par.nGridY, par.nEdge)
v = node(par, gmrf, auv)

derivXStart = auv.derivX
derivYStart = auv.derivY

fig = plt.figure()
for i in range(N):
    v.auv.x = par.x0
    v.auv.y = par.y0
    v.auv.derivX = derivXStart
    v.auv.derivY = derivYStart
    # Get first trajectory
    tau = np.zeros((2,(par.nTrajPoints-1)*depth+1))
    theta = (-1 + 2 * np.random.rand()) * np.eye(1)
    #theta = [-1]
    tau[:,0:par.nTrajPoints], v.auv.derivX, v.auv.derivY = cbts.generateTrajectory(v,theta)
    v.auv.x = tau[0, par.nTrajPoints - 1]
    v.auv.y = tau[1, par.nTrajPoints - 1]
    print("afterf first traj:",v.auv.x,v.auv.y)

    # Check next depth
    for d in range(1,depth):
        theta = (-1+2*np.random.rand())*np.eye(1)
        #theta = [-1]
        tauNext, v.auv.derivX, v.auv.derivY = cbts.generateTrajectory(v,theta)
        v.auv.x = tauNext[0,-1]
        v.auv.y = tauNext[1,-1]
        tau[:,(par.nTrajPoints-1)*d:(par.nTrajPoints-1)*(d+1)+1] = tauNext

    stepSizes = []
    xOld = par.x0
    yOld = par.y0
    for i in range(tau.shape[1]-1):
        stepSizes.append(math.sqrt((xOld-tau[0,i+1])**2+(yOld-tau[1,i+1])**2))
        xOld = tau[0,i+1]
        yOld = tau[1,i+1]
    plt.plot(tau[0,:],tau[1,:])
    plt.scatter(tau[0,:],tau[1,:])
    plt.axis('equal')
print(np.mean(stepSizes))
plt.show()
