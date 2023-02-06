import random
import math

class Module:
    def zero_grad(self):
       for p in self.parameters():
           p.grad = 0

class Value:

    def __init__(self, data, label="", inputs=[]):
        self.data = data
        self.grad = 0
        self.inputs = inputs
        self.label = label
        self._backward = lambda: None

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other, f"'{str(other)}'")
        out = Value (self.data + other.data, "", [self, other])

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __sub__(self, other):
        return self + (-1 * other)

    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        return self - other

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other, f"'{str(other)}'")
        out = Value (self.data * other.data, "", [self, other])

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        out = Value(math.tanh(self.data), "tanh", [self])

        def _backward():
            self.grad += (1 - out.data ** 2) * out.grad

        out._backward = _backward
        return out

    def  __pow__(self, k):
        out = Value(self.data ** k, "**", [self])

        def _backward():
            self.grad += k * (self.data ** (k - 1)) * out.grad

        out._backward = _backward
        return out

    def backward(self):
       self._backward()
       for i in self.inputs:
           i.backward()

    def __repr__(self):
        return f"{self.grad}|{self.data}|{self.label}|{self.inputs}"

class Neuron(Module):

    def __init__(self, nin):
        self.w = [Value(random.uniform(-1,1), f"w{str(i)}") for i in range(nin)]
        self.b = Value(0, "bias")

    def __call__(self,xs):
        res = sum([wn * xn for wn,xn in zip(self.w,xs)], self.b)
        out = res.tanh() 
        return out

    def parameters(self):
        return self.w + [self.b]

class Layer(Module):

    def __init__(self, nin, nout):
        self._neurons = [Neuron(nin) for _ in range(nout)]

    def parameters(self):
        return [p for n in self._neurons for p in n.parameters()]

    def __call__(self,xs):
        out = [n(xs) for n in self._neurons]
        return out[0] if len(out) == 1 else out
    
class MLP(Module):

    def __init__(self, nin, nouts):
        self._layers = [Layer(nin, nout) for nout in nouts]

    def __call__(self, xs):
        for layer in self._layers:
            xs = layer(xs)

        return xs

    def parameters(self):
        out = [p for l in self._layers for p in l.parameters()]
        return out[0] if len(out) == 1 else out


m = MLP(2, [12,12,1])
#xs=[[0.25, 0.25], [0.2, 0.2], [0.1, 0.1], [0.3, 0.4], [0.1, 0.4], [0.01, 0.01]]
#ys = [0.5,        0.4,        0.2,        0.7,        0.5,        0.2]

xs=[[0.25, 0.25], [0.3, 0.2], [0.8, 0.3], [0.7, 0.1], [0.9, 0.8], [0.05, 0.03]]
ys = [0,          0.1,        0.5,        0.6,        0.1,        0.03]

for _ in range(1000):
    ypred = [m(x) for x in xs]
    loss = sum([(x - y) ** 2 for (x,y) in zip(ys,ypred)])
    m.zero_grad()
    loss.grad = 1.0
    loss.backward()
    print ([y.data for y in ypred])
    print (loss.data)

    for p in m.parameters():
        p.data -= p.grad * 0.01

print ("Computing based on learned parameters")
res = m([0.25, 0.15])
print (res.data)

res = m([0.5, 0.25])
print (res.data)
