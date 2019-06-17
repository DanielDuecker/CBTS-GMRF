import numpy as np


class GP:
    def __init__(self,nTest,kernelPar,bayesianOptPar,varMeas):
        self.nTest = nTest
        self.kernelPar = kernelPar
        self.bayesianOptPar = bayesianOptPar
        self.varMeas = varMeas
        self.trainInput = None
        self.trainOutput = None

    def kernel(self,z1,z2):
        squaredDistance = np.linalg.norm(z1-z2,2)
        return np.exp(-.5 * 1/self.kernelPar * squaredDistance)

    def update(self,input,output):
        if self.trainInput == None:
            self.trainInput = input
            self.trainOutput = output
        else:
            self.trainInput = np.hstack(self.trainInput,input)
            self.trainOutput = np.hstack(self.trainOutput,output)

    def predict(self,input):
        K = self.kernel(self.trainInput,self.trainInput)

        L = np.linalg.cholesky(K + self.varMeas*np.eye(self.trainInput.shape[0]))
        alpha = np.linalg.solve(L.T,np.linalg.solve(L,self.trainOutput))












nTrain = 5
nTest = 100

