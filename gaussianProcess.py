import numpy as np

class GP:
    def __init__(self):
        self.kernelPar = 0.1

    def kernel(self,x,y):
        squaredDistance = np.sum(x**2,1)