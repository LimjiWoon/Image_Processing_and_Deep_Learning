import numpy as np
from abc import ABCMeta, abstractmethod


class AbstractActivation(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def compute(self, x):
        raise NotImplementedError()

    @abstractmethod
    def deriv(self, x):
        raise NotImplementedError()

class Loss(AbstractActivation):
    pass


class Relu(AbstractActivation):
    def compute(self, x):
        ###########################################################################
        # TODO: Implement the ReLU forward pass.                                 #
        ###########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        output = np.maximum(0, x)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

        return output

    def deriv(self, x):
        ###########################################################################
        # TODO: Implement the ReLU backward pass.                                 #
        ###########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        local_grad = 1 * (x > 0)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return local_grad

class Linear(AbstractActivation):
    def compute(self, x):
        return x

    def deriv(self, x):
        return 1.

class Sigmoid(AbstractActivation):
    def compute(self, x):
        return 1. / (1. + np.exp(-x))

    def deriv(self, x):
        y = self.compute(x)
        return y * (1. - y)

class MeanSquaredError(Loss):
    def compute(self, X, Y):
        return (1. / 2. * X.shape[0]) * ((X - Y) ** 2.)

    def deriv(self, X, Y):
        return (X - Y) / X.shape[0]

class CrossEntropy(Loss):
    def _softmax(self, X):
        expvx = np.exp(X - np.max(X, axis=1)[..., np.newaxis])
        return expvx/np.sum(expvx, axis=1, keepdims=True)

    def compute(self, X, Y):
        sf = self._softmax(X)
        return -np.log(sf[np.arange(X.shape[0]), np.argmax(Y, axis=1)]) / X.shape[0]

    def deriv(self, X, Y):
        err = self._softmax(X)
        return (err - Y) / X.shape[0]

relu = Relu()
sigmoid = Sigmoid()
linear = Linear()

mse = MeanSquaredError()
cross_entropy = CrossEntropy()
