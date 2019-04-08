import numpy as np
import matplotlib.pyplot as plt
import math

def getMeasurement(xMeas,yMeas,fGroundTruth,noiseVariance):
    noise = np.random.normal(0,math.sqrt(noiseVariance))
    return fGroundTruth(xMeas,yMeas)+noise

def mapConDis(gmrf,xMeas,yMeas,zMeas):
    # Initialize j-th row of mapping matrix phi
    phi = np.array([np.zeros(gmrf.nP),np.zeros(gmrf.nBeta)])

    # Get grid position relative to surrounding vertices
    xRel = (xMeas-gmrf.xMin) % gmrf.dx - gmrf.dx/2
    yRel = (yMeas-gmrf.yMin) % gmrf.dy - gmrf.dy/2

    # Get index of upper left neighbor 
    xPos = int((xMeas-gmrf.xMin)/gmrf.dx)
    yPos = int((yMeas-gmrf.yMin)/gmrf.dy)

    # Local coordinate system is different from Geist! (e_y=-e_y_Geist), because now mean vector is [vertice0,vertice1,vertice3,...])
    # Calculate weights at neighbouring positions
    phi[0,(yPos+1)*gmrf.nX+xPos] = 1/(gmrf.dx*gmrf.dy) * (xRel-gmrf.dx/2) * (-yRel-gmrf.dy/2)    # lower left
    phi[0,(yPos+1)*gmrf.nX+xPos+1] = -1/(gmrf.dx*gmrf.dy) * (xRel+gmrf.dx/2) * (-yRel-gmrf.dy/2) # lower right
    phi[0,yPos*gmrf.nX+xPos+1] = 1/(gmrf.dx*gmrf.dy) * (xRel+gmrf.dx/2) * (-yRel+gmrf.dy/2)      # upper right
    phi[0,yPos*gmrf.nX+xPos] = -1/(gmrf.dx*gmrf.dy) * (xRel-gmrf.dx/2) * (-yRel+gmrf.dy/2)       # upper left

    return phi  

def getPrecisionMatrix(gmrf):
    diagQ = 2*np.eye(gmrf.nP)
    diagQ[0,0] = 1
    diagQ[-1,-1] = 1
    Q = diagQ-np.eye(gmrf.nP,k=1)-np.eye(gmrf.nP,k=-1)
    return Q

def getNextState(x,y,xBefore,yBefore,maxStepsize,gmrf):

    xNext = xBefore
    yNext = yBefore

    stepsize = maxStepsize*np.random.rand()

    # avoid going back to former location:
    while(xNext == xBefore and yNext == yBefore):
        xNext = np.random.choice([x-stepsize,x+stepsize])
        yNext = np.random.choice([y-stepsize,y+stepsize])

    if xNext < gmrf.xMin:
        xNext = x+stepsize
    elif xNext > gmrf.xMax:
        xNext = x-stepsize
    
    if yNext < gmrf.yMin:
        yNext = y+stepsize
    elif yNext > gmrf.yMax:
        yNext = y-stepsize
    
    return (xNext,yNext)

def plotFields(fig,x,y,f,gmrf,iterVec,timeVec,xHist,yHist):
    plt.clf()

    # Plotting ground truth
    plt.ion()
    ax1 = fig.add_subplot(221)
    ax1.contourf(x,y,f(x,y))
    plt.title("True field") 

    # Plotting conditioned mean
    ax2 = fig.add_subplot(222)
    ax2.contourf(gmrf.x,gmrf.y,gmrf.meanCond[0:nP].reshape(gmrf.nY,gmrf.nX))
    plt.xlabel("x in m")
    plt.ylabel("y in m")
    plt.title("Mean of belief")

    # Plotting precision matrix
    ax3 = fig.add_subplot(223)
    ax3.contourf(gmrf.x,gmrf.y,np.diag(gmrf.covCond[0:nY,0:nX]))
    ax3.plot(xHist,yHist,'black')
    plt.xlabel("x in m")
    plt.ylabel("y in m")
    plt.title("Precision of belief")

    # Plotting time consumption
    ax4 = fig.add_subplot(224)
    ax4.plot(iterVec,timeVec,'black')
    plt.xlabel("Iteration index")
    plt.ylabel("calculation time in s")
    plt.title("Update calculation time over iteration index")

    fig.canvas.draw()