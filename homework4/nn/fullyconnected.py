from nn.layer import AbstractLayer
from nn.optimizer import SGDOptimizer

class FullyConnected(AbstractLayer):
    def __init__(self, wshape, activation, weight_init, optimizer = SGDOptimizer()):
        self.wshape = wshape
        self.w = weight_init(self.wshape)
        self.activation = activation
        self.optimizer = optimizer

    def forward(self, inputs):
        z = inputs.dot(self.w)
        return (z, self.activation.compute(z))

    def get_activation_grad(self, z, upstream_gradient):
        return upstream_gradient * self.activation.deriv(z)

    def backward(self, layer_err):
        return layer_err.dot(self.w.T)

    def get_grad(self, inputs, layer_err):
        return inputs.T.dot(layer_err)

    def update(self, grad, lr):
        self.w -= self.optimizer.update(grad, lr)

