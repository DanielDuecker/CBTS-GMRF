import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
matplotlib.use('TkAgg')

class GP:
    def __init__(self,kernelPar,varMeas):
        self.kernelPar = kernelPar
        self.varMeas = varMeas
        self.emptyData = True

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

    def update(self,inputData,outputData):
        if self.emptyData:
            self.trainInput = inputData
            self.trainOutput = outputData
            self.emptyData = False
        else:
            self.trainInput = np.vstack((self.trainInput,inputData))
            self.trainOutput = np.vstack((self.trainOutput,outputData))

    def predict(self,input):
        # mu = K(test,training).T*inv(K(training,training))*trainingOutput
        K = self.getKernelMatrix(self.trainInput,self.trainInput)
        L = np.linalg.cholesky(K + self.varMeas*np.eye(self.trainInput.shape[0]))

        # Compute mean
        Lk = np.linalg.solve(L,self.getKernelMatrix(self.trainInput,input))
        mu = np.dot(Lk.T, np.linalg.solve(L,self.trainOutput))

        # Compute variance
        KStar = self.getKernelMatrix(input,input)
        var = KStar - np.dot(Lk.T,Lk)

        return mu, var

# Parameter
kernelPar = 1
varMeas = 0.001
kappa = 100
GP = GP(kernelPar,varMeas)

# Ground Truth
#f = lambda x,y: x**2 + 0.9*y**2
f = lambda x,y: (np.sin(x) + np.sin(y))*np.exp(-0.1*np.abs(x+y))
xGT0, xGT1 = np.meshgrid(np.linspace(-5,5,100),np.linspace(-5,5,100))
fGT = f(xGT0,xGT1)
#print("fGT:",fGT)

xTrain = np.random.uniform(-5,5,(1,2))
xTrainHist = np.zeros((1000,2))
fTrainHist = np.zeros((1000,1))

fig = plt.figure()
plt.ion()
plt.show()
for i in range(100):
    print(i)
    # next measurement:
    fTrain = f(xTrain[:,0],xTrain[:,1]) + varMeas*np.random.randn()
    fTrain = fTrain.reshape(-1,1)
    GP.update(xTrain,fTrain)

    nSample = 100
    xSample = np.random.uniform(-5,5,(nSample,2))
    mu,var = GP.predict(xSample)
    xTrainHist[i,:] = xTrain
    fTrainHist[i] = fTrain

    # acquisition function
    H = mu.reshape(nSample,1) + kappa*np.sqrt(var.diagonal()).reshape(nSample,1)
    index = np.argmax(H)
    xTrain = xSample[index,:].reshape(1,2)

    if i%10 == 0:
        ax = fig.add_subplot(111,projection='3d')
        ax.plot_wireframe(xGT0, xGT1, fGT)
        ax.plot(xTrainHist[:,0],xTrainHist[:,1],fTrainHist[:,0],"g.")
        ax.plot(xSample[:,0],xSample[:,1],mu[:,0],"r.")
        plt.title("True field")
        print("difference:",np.mean(mu-f(xSample[:,0],xSample[:,1])))
        fig.canvas.draw()

plt.show(block=True)


