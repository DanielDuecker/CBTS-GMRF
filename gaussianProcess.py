import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import  Axes3D


class GP:
    def __init__(self,kernelPar,varMeas):
        self.kernelPar = kernelPar
        self.varMeas = varMeas
        self.init = False

    def kernel(self,z1,z2):
        squaredDistance = np.linalg.norm(z1-z2,2)
        return np.exp(-.5 * 1/self.kernelPar * squaredDistance)

    def getKernelMatrix(self,vec1,vec2):
        n = vec1.shape[0]
        N = vec2.shape[0]
        K = np.zeros((n,N))
        for i in range(n):
            for j in range(N):
                 K[i,j] = self.kernel(vec1[i,:],vec2[j,:])
        return K
        # todo: only update K matrix instead of recalculating

    def update(self,input,output):
        if self.init == False:
            self.trainInput = input
            self.trainOutput = output
            self.init = True
        else:
            self.trainInput = np.vstack((self.trainInput,input))
            self.trainOutput = np.vstack((self.trainOutput,output))

    def predict(self,input):
        K = self.getKernelMatrix(self.trainInput,self.trainInput)
        L = np.linalg.cholesky(K + self.varMeas*np.eye(self.trainInput.shape[0]))

        # Compute mean
        Lk = np.linalg.solve(L,self.getKernelMatrix(self.trainInput,input))
        mu = np.dot(Lk.T, np.linalg.solve(L,self.trainOutput))

        #print("input:",input)
        print("mu:",mu)

        # Compute variance
        KStar = self.getKernelMatrix(input,input)
        var = KStar - np.dot(Lk.T,Lk)

        return mu, var

# Parameter
kernelPar = 0.1
varMeas = 0.1
kappa = 1
GP = GP(kernelPar,varMeas)

# Ground Truth
f = lambda x,y: x**2 + 0.9*y**2
#f = lambda x,y: np.sin(x) + np.sin(y)
xGT0, xGT1 = np.meshgrid(np.linspace(-5,5,100),np.linspace(-5,5,100))
fGT = f(xGT0,xGT1)
#print("fGT:",fGT)

xTrain = np.random.uniform(-5,5,(1,2))
xTrainHist = np.zeros((1000,2))
fTrainHist = np.zeros((1000,1))

for i in range(1000):
    print(i)
    # next measurement:
    fTrain = f(xTrain[:,0],xTrain[:,1]) + varMeas*np.random.randn()
    GP.update(xTrain,fTrain)

    nSample = 20
    xSample = np.random.uniform(-5,5,(nSample,2))
    mu,var = GP.predict(xSample)
    xTrainHist[i,:] = xTrain
    fTrainHist[i] = fTrain

    # acquisition function
    # todo size of mu changes sometimes. Why?
    H = mu.reshape(nSample,1) + kappa*var.diagonal().reshape(nSample,1)
    index = np.argmax(H)
    xTrain = xSample[index,:].reshape(1,2)
    #print(H)
    #print(index)

    fig = plt.figure()

    ax = fig.add_subplot(111,projection='3d')
    ax.plot_surface(xGT0, xGT1, fGT)
    ax.plot(xTrainHist[:,0],xTrainHist[:,1],fTrainHist[:,0],"g.")
    print(mu.shape)
    ax.plot(xSample[:,0],xSample[:,1],mu.reshape(20,1),"r.")
    plt.title("True field")
    print(xSample[:,0].shape)
    print(xSample[:,1].shape)
    print("difference:",mu-f(xSample[:,0],xSample[:,1]))

plt.show()


