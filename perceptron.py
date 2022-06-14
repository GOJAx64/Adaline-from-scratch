import numpy as np

class Perceptron:
    def __init__(self, n, learnign_rate):
        self.w = -1 + 2 * np.random.rand(n)
        self.b = -1 + 2 * np.random.rand()
        self.eta = learnign_rate
        self.converged = False
        self.epochs = 0

    def pw(self, z):
        return 1 if z >= 0 else 0

    def net(self, X):
        return np.dot(self.w, X) + self.b

    def fitness(self, X, Y, epochs):
        done = False
        epoch = 0
        p = X.shape[1]

        while(not done and epoch < epochs):
            done = True
            for i in range(p):
                error = Y[i] - self.pw(self.net(X[:,i]))
                if error != 0:
                    done = False
                    self.w += self.eta * error * X[:,i]
                    self.b += self.eta * error
            epoch += 1
            
        self.converged = done
        self.epochs = epoch

    def get_w(self):
        return self.w

    def get_b(self):
        return self.b

    def hasConverged(self):
        return self.converged
    
    def epochsReached(self):
        return self.epochs