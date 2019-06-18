import numpy as np
import copy
import parameters as par

class node:
    def __init__(self,gmrf,auv,r):
        self.gmrf = copy.deepcopy(gmrf)
        self.auv = copy.deepcopy(auv)
        self.totalR = copy.deepcopy(r)
        self.actionToNode = []
        self.depth = 0
        self.parent = []
        self.children = []
        self.visits = 1
        self.D = []
        self.GP = GP()

class GP:
    def __init__(self):
        self.emptyData = True
        self.trainInput = None
        self.trainOutput = None

    def kernel(self,z1,z2):
        squaredDistance = np.linalg.norm(z1-z2,2)
        return np.exp(-.5 * 1/par.kernelPar * squaredDistance)

    def getKernelMatrix(self,vec1,vec2):
        print(vec1)
        print(vec2)
        n = vec1.shape[0]
        N = vec2.shape[0]
        K = np.zeros((n,N))
        for i in range(n):
            for j in range(N):
                 K[i,j] = self.kernel(vec1[i,:],vec2[j,:])
        return K

    def update(self,inputData,outputData):
        if self.emptyData:
            self.trainInput = inputData
            self.trainOutput = outputData
            self.emptyData = False
        else:
            self.trainInput = np.vstack((self.trainInput,inputData))
            self.trainOutput = np.vstack((self.trainOutput,outputData))

    def predict(self,input):
        # according to https://www.cs.ubc.ca/~nando/540-2013/lectures/l6.pdf
        K = self.getKernelMatrix(self.trainInput,self.trainInput)
        L = np.linalg.cholesky(K)

        # Compute mean
        Lk = np.linalg.solve(L,self.getKernelMatrix(self.trainInput,input))
        mu = np.dot(Lk.T, np.linalg.solve(L,self.trainOutput))

        # Compute variance
        KStar = self.getKernelMatrix(input,input)
        var = KStar - np.dot(Lk.T,Lk)

        return mu, var


