import numpy as np

class Node:
    def __init__(self, inputs=[], name=None, is_trainable=True):
        self.inputs = inputs
        self.outputs = []
        self.name = name
        self.is_trainable = is_trainable

        for n in self.inputs:
            n.outputs.append(self)

        self.value = None
        self.gradients = {}

    def forward(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError

    def __repr__(self):
        return self.name


class Placeholder(Node):
    def __init__(self, name, is_trainable=True):
        Node.__init__(self, name=name, is_trainable=is_trainable)

    def forward(self, value=None):
        if value is not None: self.value = value

    def backward(self):
        # input N --> N1, N2
        # partial L / partial N ==> partial L / partial N1 * partial N1 / partial N
        # where partial N1 / partial N = 1
        self.gradients = {}
        for n in self.outputs:
            self.gradients[self] = n.gradients[self]

    def __repr__(self):
        return 'Placeholer: {}'.format(self.name)


class Add(Node):
    def __init__(self, *nodes):
        Node.__init__(self, [nodes])

    def forward(self):
        # value = sum(inputs.value)
        self.value = sum(map(lambda n: n.value, self.inputs))


class Linear(Node):
    def __init__(self, x=None, weigth=None, bias=None, name=None, is_trainable=False):
        Node.__init__(self, [x, weigth, bias], name=name, is_trainable=is_trainable)

    def forward(self):
        k, x, b = self.inputs[1], self.inputs[0], self.inputs[2]
        self.value = x.value * k.value + b.value

    def backward(self):
        self.gradients = {n: np.zeros_like(n.value) for n in self.inputs}
        k, x, b = self.inputs[1], self.inputs[0], self.inputs[2]

        for n in self.outputs:
            grad_cost = n.gradients[self]
            self.gradients[self.inputs[0]] = np.dot(grad_cost, k.value.T)
            self.gradients[self.inputs[1]] = np.dot(x.value.T, grad_cost)
            self.gradients[self.inputs[2]] = np.sum(grad_cost, axis=0, keepdims=False)

    def __repr__(self):
        return 'Linear: {}'.format(self.name)


class Sigmoid(Node):
    def __init__(self, x, name=None, is_trainable=False):
        Node.__init__(self, [x], name=name, is_trainable=is_trainable)
        self.x = self.inputs[0]

    def _sigmoid(self, x):
        return 1./(1 + np.exp(-1 * x))

    def forward(self):
        self.value = self._sigmoid(self.x.value)

    def partial(self):
        return self._sigmoid(self.x.value) * (1 - self._sigmoid(self.x.value))

    def backward(self):
        self.gradients = {n: np.zeros_like(n.value) for n in self.inputs}
        for n in self.outputs:
            grad_cost = n.gradients[self]
            self.gradients[self.x] = grad_cost * self.partial()
            # print(self.gradients)


class Relu(Node):
    def __init__(self, x, name=None, is_trainable=False):
        Node.__init__(self, [x], name=name, is_trainable=is_trainable)
        self.x = x

    def forward(self):
        self.value = self.x.value * (self.x.value > 0)

    def backward(self):
        for n in self.outputs:
            grad_cost = n.gradients[self]
            self.gradients[self.x] = grad_cost * (self.x.value > 0)


class L2_LOSS(Node):
    def __init__(self, y, y_hat, name=None, is_trainable=False):
        Node.__init__(self, [y, y_hat], name=name, is_trainable=is_trainable)
        self.y = y
        self.y_hat = y_hat

    def forward(self):
        # assert (self.y.shape == self.y_hat.shape)
        # self.size = self.inputs[0].value.shape[0]
        self.diff = self.y.value - self.y_hat.value
        self.value = np.mean(self.diff ** 2)

    def backward(self):
        # 1/n sum (y- yhat)**2
        self.gradients[self.y] = 2 * np.mean((self.y.value - self.y_hat.value))
        self.gradients[self.y_hat] = -2 * np.mean((self.y.value - self.y_hat.value))

def optimize(graph, learning_rate=1e-3):
    # there are so many other update / optimization methods
    # such as Adam, Mom,
    for t in graph:
        if t.is_trainable:
            t.value += -1 * learning_rate * t.gradients[t]