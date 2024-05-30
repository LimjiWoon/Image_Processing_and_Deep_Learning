import numpy as np
from collections import deque

class Network(object):
    def __init__(self, layers, lr, loss):
        self.layers = layers
        self.loss = loss
        self.lr = lr

    def train_step(self, mini_batch):
        mini_batch_inputs, mini_batch_outputs = mini_batch

        # 스택 자료구조를 이용한 뉴럴넷 레이어 구조
        # forward pass
        z_stack = deque([mini_batch_inputs])
        activation = mini_batch_inputs
        for l in self.layers:
            z, activation = l.forward(activation)
            z_stack.append(z)
            z_stack.append(activation)

        # calculate loss
        la = z_stack.pop()
        loss_err = self.loss.deriv(la, mini_batch_outputs) # p-z

        # backward pass
        upstream_gradient = loss_err
        grads = deque()
        for l in reversed(self.layers):
            lz = z_stack.pop()
            layer_err = l.get_activation_grad(lz, upstream_gradient)

            la = z_stack.pop()
            grads.append(l.get_grad(la, layer_err))
            upstream_gradient = l.backward(layer_err) # downstream grad

        # update step
        for l in self.layers:
            l.update(grads.pop(), self.lr)

        assert len(grads) == 0

    # forward pass
    def forward(self, inputs):
        activation = inputs
        for l in self.layers:
            z, activation = l.forward(activation)
        return activation