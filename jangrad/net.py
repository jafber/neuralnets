from random import uniform
from value import Value
from typing import List


class Neuron:

    def __init__(self, n_inputs: int, nonlin: bool = True):
        self.weights = [Value(uniform(-1.0, 1.0), label='w')
                        for _ in range(n_inputs)]
        self.bias = Value(0.0, label='b')
        self.nonlin = nonlin

    def __call__(self, inputs: List[Value]) -> Value:
        assert len(inputs) == len(self.weights)
        activation = self.bias
        # idk why but using sum adds a Value(0.0), so we just do a loop
        for weight, input in zip(self.weights, inputs):
            activation += weight * input
        return activation.relu() if self.nonlin else activation

    def params(self) -> List[Value]:
        return self.weights + [self.bias]


class Layer:

    def __init__(self, n_inputs: int, n_neurons: int, nonlin: bool):
        self.n_inputs = n_inputs
        self.neurons = [Neuron(n_inputs, nonlin) for _ in range(n_neurons)]

    def __call__(self, inputs: List[Value]) -> List[Value]:
        assert len(inputs) == self.n_inputs
        return [n(inputs) for n in self.neurons]

    def params(self) -> List[Value]:
        return [param for neuron in self.neurons for param in neuron.params()]


class MLP:
    # multi-layer perceptron

    def __init__(self, n_inputs, layer_sizes: List[int]):
        self.n_inputs = n_inputs
        self.layers = []
        for i in range(len(layer_sizes)):
            if i == 0:
                layer = Layer(n_inputs, layer_sizes[i], True)
            elif i == len(layer_sizes) - 1:
                layer = Layer(layer_sizes[i-1], layer_sizes[i], False)
            else:
                layer = Layer(layer_sizes[i-1], layer_sizes[i], True)
            self.layers.append(layer)

    def __call__(self, input: List[Value]) -> List[Value]:
        assert len(input) == self.n_inputs
        for layer in self.layers:
            input = layer(input)
        return input

    def loss(self, input: List[Value], desired: List[Value] | Value) -> Value:
        if not isinstance(desired, list):
            desired = [desired]
        assert len(input) == self.n_inputs
        output = self(input)
        assert len(desired) == len(output)
        return sum(
            [(out_val - des_val) ** 2 for out_val, des_val in zip(output, desired)]
        )

    def params(self) -> List[Value]:
        return [param for layer in self.layers for param in layer.params()]

    def learn(self, ins: List[List[Value]], outs: List[List[Value]], learning_rate: float) -> float:
        assert len(ins) == len(outs)
        total_loss = sum(
            [self.loss(input, desired) for input, desired in zip(ins, outs)]
        )
        total_loss.backward()
        for param in self.params():
            # param.grad points in the direction of increased loss
            # it tells us how much loss increases when we increase param.grad
            param.data -= learning_rate * param.grad
        return total_loss
