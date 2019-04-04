import numpy as np
import math

def getMeasurement(xMeas,yMeas,fGroundTruth,noiseVariance):
    noise = np.random.normal(0,math.sqrt(noiseVariance))
    return fGroundTruth(xMeas,yMeas)+noise

def mapConDis(gmrf,xMeas,yMeas,zMeas):
    # Initialize j-th row of mapping matrix phi
    phi = np.array([np.zeros(gmrf.nP)])

    # Get grid position relative to surrounding vertices
    xRel = (xMeas-gmrf.xMin) % gmrf.dx - gmrf.dx/2
    yRel = (yMeas-gmrf.yMin) % gmrf.dy - gmrf.dy/2

    # Get index of Neighbor 
    xPos = int((xMeas-xRel)/gmrf.dx)
    yPos = int((yMeas-yRel)/gmrf.dy)

    #DIFFERENT FROM GEIST!
    # Local coordinate system is different from Geist! (e_y=-e_y_Geist)
    # Calculate weights at neighbouring positions
    phi[0,(yPos+1)*gmrf.nX+xPos] = 1/(gmrf.dx*gmrf.dy) * (xRel-gmrf.dx/2) * (-yRel-gmrf.dy/2)    # lower left
    phi[0,(yPos+1)*gmrf.nX+xPos+1] = -1/(gmrf.dx*gmrf.dy) * (xRel+gmrf.dx/2) * (-yRel-gmrf.dy/2) # lower right
    phi[0,yPos*gmrf.nX+xPos+1] = 1/(gmrf.dx*gmrf.dy) * (xRel+gmrf.dx/2) * (-yRel+gmrf.dy/2)      # upper right
    phi[0,yPos*gmrf.nX+xPos] = -1/(gmrf.dx*gmrf.dy) * (xRel-gmrf.dx/2) * (-yRel+gmrf.dy/2)       # upper left

    # DEBUGGING
    #print("dx:",gmrf.dx)
    #print("dy:",gmrf.dy)
    #print("x:",gmrf.x)
    #print("_____")
    #print("Meas: (",xMeas,",",yMeas,")")
    #print("zMeas:",zMeas[-1])
    #print("Rel: (",xRel,",",yRel,")")
    #print("Pos: (",xPos,",",yPos,")")
    #print("lower left:",(yPos+1)*gmrf.nX+xPos)
    #print("lower right:",(yPos+1)*gmrf.nX+xPos+1)
    #print("upper left:",yPos*gmrf.nX+xPos)
    #print("upper right:",yPos*gmrf.nX+xPos+1)
    #print(phi)

    return phi  

def getPrecisionMatrix(gmrf):
    Q = np.eye(gmrf.nP)+0.1*np.eye(gmrf.nP,k=1)+0.1*np.eye(gmrf.nP,k=-1)
    return Q


