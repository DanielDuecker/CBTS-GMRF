import parameters as par
import numpy as np
from kCBTS import mapActionReward

map = mapActionReward(-1,2,3,3)
print(map.meanCond)
print(map.covCond)

print("First update:")
map.updateMapActionReward(np.array([[0.1],[0.2],[0.3]]),100)
print(map.meanCond)
print(map.covCond)

print("Second update:")
map.updateMapActionReward(np.array([[0],[0.9],[0.8]]),-200)

print("Test on theta=[0.5 0.5 0.5]")
thetaTest = np.array([[0.5],[0.5],[0.5]])
print("Cov function:")
print(map.getCovarianceFromAction(thetaTest))

print("Get best theta:")
print(map.getBestTheta())