import numpy as np
from nn.layer import AbstractLayer

class MaxPooling(AbstractLayer):
    def __init__(self, pshape, strides=1):
        self.pshape = pshape  # pooling shape (height * width)
        self.strides = strides
        self.cached_data = []

    ###########################################################################
    # TODO: Implement the Max-pooling layer                                   #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    def forward(self, inputs):
        pass
        return None

    def get_activation_grad(self, z, upstream_gradient):
        # There is no activation function
        return upstream_gradient

    def backward(self, layer_err):
        pass
        return None

    def get_grad(self, inputs, layer_err):
        pass
        return None

    def update(self, grad, lr):
        pass
        return None

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################