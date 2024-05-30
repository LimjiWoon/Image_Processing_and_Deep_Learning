import numpy as np
import math
from nn.layer import AbstractLayer
from nn.optimizer import SGDOptimizer

class Conv(AbstractLayer):
    def __init__(self, fshape, activation, filter_init, optimizer = SGDOptimizer(), strides=1, pad=0):
        self.fshape = fshape
        self.strides = strides
        self.filters = filter_init(self.fshape)
        self.activation = activation
        self.optimizer = optimizer

    def forward(self, inputs):
        """
        A naive implementation of the forward pass for a convolutional layer.

        The input consists of N data points, each with C channels, height H and
        width W. We convolve each input with F different filters, where each filter
        spans all C channels and has height HH and width WW.

        Input:
        - x: Input data of shape (N, H, W, C)

        Attributes:
        - w: Filter weights of shape (HH, WW, C, F)
        - 'stride': The number of pixels between adjacent receptive fields in the
            horizontal and vertical directions.

        Returns a tuple of:
        - out: feature map, of shape (N, H', W', F) where H' and W' are given by
          H' = 1 + (H - HH) / stride
          W' = 1 + (W - WW) / stride
        """
        s = int((inputs.shape[1] - self.fshape[0]) / self.strides + 1)
        feature_map = np.zeros((inputs.shape[0], s, s, self.fshape[-1]))
        for j in range(s):
            for i in range(s):
                feature_map[:, j, i, :] = np.sum(inputs[:, j * self.strides:j * self.strides + self.fshape[0], i * self.strides:i * self.strides + self.fshape[1], :, np.newaxis] * self.filters, axis=(1, 2, 3))
        return (feature_map, self.activation.compute(feature_map))

    def get_activation_grad(self, z, upstream_gradient):
        return upstream_gradient * self.activation.deriv(z)

    def backward(self, layer_err):
        bfmap_shape = (layer_err.shape[1] - 1) * self.strides + self.fshape[0]
        backwarded_fmap = np.zeros((layer_err.shape[0], bfmap_shape, bfmap_shape, self.fshape[-2]))
        s = int((backwarded_fmap.shape[1] - self.fshape[0]) / self.strides + 1)
        for j in range(s):
            for i in range(s):
                backwarded_fmap[:, j * self.strides:j  * self.strides + self.fshape[0], i * self.strides:i * self.strides + self.fshape[1]] += \
                    np.sum(self.filters[np.newaxis, ...] * layer_err[:, j:j+1, i:i+1, np.newaxis, :], axis=4)
        return backwarded_fmap

    def get_grad(self, inputs, layer_err):
        total_layer_err = np.sum(layer_err, axis=(0, 1, 2))
        filters_err = np.zeros(self.fshape)
        s = int((inputs.shape[1] - self.fshape[0]) / self.strides + 1)
        summed_inputs = np.sum(inputs, axis=0)
        for j in range(s):
            for i in range(s):
                filters_err += summed_inputs[j  * self.strides:j * self.strides + self.fshape[0], i * self.strides:i * self.strides + self.fshape[1], :, np.newaxis]
        return filters_err * total_layer_err

    def update(self, grad, lr):
        self.filters -= self.optimizer.update(grad, lr)