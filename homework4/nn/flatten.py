import numpy as np
from nn.layer import AbstractLayer

class Flatten(AbstractLayer):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, inputs):
        z = np.reshape(inputs, (inputs.shape[0], -1))
        return (z, z)

    def get_activation_grad(self, z, upstream_gradient):
        # There is no activation function
        return upstream_gradient

    def backward(self, layer_err):
        return np.reshape(layer_err, (layer_err.shape[0],) + self.shape)

    def get_grad(self, inputs, layer_err):
        return 0.

    def update(self, grad, lr):
        pass
