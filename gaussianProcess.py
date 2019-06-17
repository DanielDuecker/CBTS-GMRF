import numpy as np
import matplotlib.pyplot as plt


class GP:
    def __init__(self,kernelPar,varMeas):
        self.kernelPar = kernelPar
        self.varMeas = varMeas
        self.trainInput = None
        self.trainOutput = None

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
        if self.trainInput == None:
            self.trainInput = input
            self.trainOutput = output
        else:
            self.trainInput = np.vstack(self.trainInput,input)
            self.trainOutput = np.vstack(self.trainOutput,output)

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
kernelPar = 10
varMeas = 0.001
kappa = 1
GP = GP(kernelPar,varMeas)

# Ground Truth
f = lambda x,y: x**2 + 0.9*y**2
xGT0, xGT1 = np.meshgrid(np.linspace(-5,5,1000),np.linspace(-5,5,1000))
fGT = f(xGT0,xGT1)
#print("fGT:",fGT)

# Training data
nTrain = 100
xTrain = np.random.uniform(-5,5,(nTrain,2))
fTrain = f(xTrain[:,0],xTrain[:,1]) + varMeas*np.random.randn(nTrain)
GP.update(xTrain,fTrain)

nSample = 10
xSample = np.random.uniform(-5,5,(nSample,2))
mu,var = GP.predict(xSample)

# acquisition function
H = mu + kappa*var.diagonal()
index = np.argmax(H)

fig = plt.figure()

ax1 = fig.add_subplot(221)
ax1.contourf(xGT0, xGT1, fGT)
ax1.plot(xTrain[:,0],xTrain[:,1],"g.")
ax1.plot(xSample[:,0],xSample[:,1],"r.")
plt.title("True field")

print("difference:",mu-f(xSample[:,0],xSample[:,1]))

#ax2 = fig.add_subplot(222)
#ax2.contourf(xSample[0,:],xSample[1,:],mu)

plt.show()


