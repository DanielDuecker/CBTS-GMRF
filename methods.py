import numpy as np
import math

def getMeasurement(xMeas,yMeas,fGroundTruth,noiseVariance):
    noise = np.random.normal(0,math.sqrt(noiseVariance))
    return fGroundTruth(xMeas,yMeas)+noise

def mapConDis(gmrf,xMeas,yMeas,zMeas):
    # Initialize j-th row of mapping matrix phi
    phi = np.array([np.zeros(gmrf.nP)])

    # Get grid position relative to surrounding vertices
    xRel = (xMeas-gmrf.xMin) % gmrf.dx
    yRel = (yMeas-gmrf.yMin) % gmrf.dy

    # Get index of Neighbors 
    xPos = int((xMeas-xRel)/gmrf.dx)
    yPos = int((yMeas-yRel)/gmrf.dy)

    #DIFFERENT FROM GEIST!
    # Calculate weights at neighbouring positions
    phi[0,(yPos+1)*gmrf.nX+xPos] = 1/(gmrf.dx*gmrf.dy) * (xRel-gmrf.dx/2) * (yRel-gmrf.dy/2)    # lower left
    phi[0,(yPos+1)*gmrf.nX+xPos+1] = -1/(gmrf.dx*gmrf.dy) * (xRel+gmrf.dx/2) * (yRel-gmrf.dy/2) # lower right
    phi[0,yPos*gmrf.nX+xPos] = 1/(gmrf.dx*gmrf.dy) * (xRel+gmrf.dx/2) * (yRel+gmrf.dy/2)       # upper left
    phi[0,yPos*gmrf.nX+xPos+1] = -1/(gmrf.dx*gmrf.dy) * (xRel-gmrf.dx/2) * (yRel+gmrf.dy/2)        # upper right

    return phi  



