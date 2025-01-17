from typing import Any, Self
from collections import deque
from statistics import NormalDist  # i love the python stdlib <3


class Value:

    # in binary operators, cast the other argument to Value if it is a float
    @staticmethod
    def _cast_other(func):
        def wrap(self: Self, other: Any) -> Self:
            other = other if isinstance(other, Value) else Value(other)
            return func(self, other)
        return wrap

    def __init__(self, data, op=None, children=None, label=None, grader=None):
        self.data = data
        self.op = op
        self.grad = 0.0
        self.children = children if children else []
        self.label = label
        self.grader = grader if grader else (lambda arg: None)

    def __repr__(self) -> str:
        return f"{self.data:.2f}"

    @_cast_other
    def __add__(self, other: Self) -> Self:
        # f(a, b) = a + b; df/da = 1.0; df/db = 1.0
        def grade(parent: Self) -> None:
            # we can just add gradients on because of chain rule
            self.grad += 1.0 * parent.grad
            other.grad += 1.0 * parent.grad
        return Value(self.data + other.data, '+', [self, other], grader=grade)

    def __neg__(self) -> Self:
        # f(a) = -a; df/da = -1.0
        def grade(parent: Self):
            # multiply with parent.grad bc chain rule
            # += bc of addition rule, add to gradients from other successors
            self.grad += -1.0 * parent.grad
        return Value(-self.data, '-', [self], grader=grade)

    @_cast_other
    def __radd__(self, other: Self) -> Self:
        return self.__add__(other)

    @_cast_other
    def __sub__(self, other: Self) -> Self:
        return self.__add__(other.__neg__())

    @_cast_other
    def __rsub__(self, other: Self) -> Self:
        return other.__add__(self.__neg__())

    @_cast_other
    def __mul__(self, other: Self) -> Self:
        # f(a, b) = a * b; df/da = b; df/db = a
        def grade(parent: Self) -> None:
            self.grad += other.data * parent.grad
            other.grad += self.data * parent.grad
        return Value(self.data * other.data, '*', [self, other], grader=grade)

    @_cast_other
    def __rmul__(self, other: Self) -> Self:
        return self.__mul__(other)

    @_cast_other
    def __pow__(self, other: Self) -> Self:
        # f(a, b) = a ** b; df/da = b a**(b-1); df/db = ln(a) a**b
        def grade(parent: Self) -> None:
            self.grad += other.data * self.data**(other.data-1) * parent.grad
            # apparently doing fractional powers of negative numbers can produce complex numbers???
            # like the normal ones aren't already complex enough
            # other.grad += log(self.data) * (self.data ** other.data) * parent.grad
            assert(not other.children)
        return Value(self.data ** other.data, '**', [self, other], grader=grade)

    # gaussian-error linear unit
    def gelu(self) -> Self:
        # https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
        def grade(parent: Self) -> None:
            deriv = self.data * NormalDist().pdf(self.data) + NormalDist().cdf(self.data)
            self.grad += deriv * parent.grad
        val = self.data * NormalDist().cdf(self.data)
        return Value(val, 'gelu', [self], grader=grade)

    # rectified linear unit activation function
    def relu(self) -> Self:
        def grade(parent: Self) -> None:
            self.grad += parent.grad * (1.0 if self.data > 0.0 else 0.0)
        return Value(max(0, self.data), 'relu', [self], grader=grade)

    def zero(self) -> None:
        self.grad = 0.0
        for child in self.children:
            child.zero()

    def backward(self) -> None:
        # before we apply gradient to children, zero old gradients
        self.zero()
        self.grad = 1.0

        # bfs to make sure all parents have graded a node before it grades it's children
        nodes = deque([self])
        vis = set()
        while nodes:
            node = nodes.pop()
            # only push each node's gradient onto its children once
            if not node in vis:
                node.grader(node)
                vis.add(node)
                nodes.extend(node.children)
