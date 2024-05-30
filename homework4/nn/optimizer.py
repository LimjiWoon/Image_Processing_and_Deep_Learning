import numpy as np

class SGDOptimizer():
    def __init__(self):
        pass

    def update(self, dx, lr = 0.001):
        update_value = dx * lr
        return update_value




class AdamOptimizer():
    def __init__(self):
        # parameters for Adam
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.m = 0.0
        self.v = 0.0
        self.eps = 1e-8
        self.t = 0

    def update(self, dx, lr = 0.001):
        """
        A implementation of the Adam optimizer.

        Input:
        - dx: gradient of the target weight
        - lr: learning rate

        Returns a tuple of:
        - out: update_value
        """
        ###########################################################################
        # TODO: Implement the Adam optimizer                                      #
        ###########################################################################
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * dx
        self.v = self.beta2 * self.v + (1 - self.beta2) * dx * dx
        m_unbias = self.m / (1 - self.beta1 ** self.t)
        v_unbias = self.v / (1 - self.beta2 ** self.t)
        update_value = lr * m_unbias / (np.sqrt(v_unbias + self.eps))

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return update_value


class MomentumSGDOptimizer():
    def __init__(self):
        # parameters for Adam
        self.beta = 0.9
        self.v = 0

    def update(self, dx, lr = 0.001):
        """
        A implementation of the Momentum SGD optimizer.

        Input:
        - dx: gradient of the target weight
        - lr: learning rate

        Returns a tuple of:
        - out: update_value
        """
        ###########################################################################
        # TODO: Implement the Momentum SGD optimizer                              #
        ###########################################################################

        self.v = self.beta * self.v + dx
        update_value = lr * self.v

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return update_value
