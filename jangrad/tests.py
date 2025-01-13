import unittest
from value import Value as V
from net import Neuron as N


class ValueTest(unittest.TestCase):
    def test_something(self):
        a = V(1.0)
        self.assertEqual(a.data, 1.0)

    def test_add(self):
        a = V(1.0) + 2.0
        self.assertEqual(a.data, 3.0)
        b = 2.0 + V(1.0)
        self.assertEqual(b.data, 3.0)
        c = V(1.0) + V(2.0)
        self.assertEqual(c.data, 3.0)

    def test_sub(self):
        cases = [
            V(1.0) - 2.0,
            1.0 - V(2.0),
            V(1.0) - V(2.0),
        ]
        for case in cases:
            self.assertEqual(-1.0, case.data)

    def test_pow(self):
        a = V(3.0)
        b = V(2.0)
        c = (a**b) * 2.0
        # dc/da = 2 b a**(b-1); dc/db = 2 ln(a) a**b
        c.backward()
        # 3**2 * 2 = 18
        self.assertEqual(c.data, 18)
        # 2 b a**(b-1) = 12
        self.assertEqual(a.grad, 12.0)
        # 2 ln(a) a**b = 18 ln(3) ~~ 19.7750212
        # self.assertAlmostEqual(b.grad, 19.7750212)

    def test_relu(self):
        a = V(3.2)
        b = a.relu()
        b.backward()
        self.assertEqual(b.data, 3.2)
        self.assertEqual(a.grad, 1.0)

        c = V(-3.2)
        d = c.relu()
        d.backward()
        self.assertEqual(d.data, 0.0)
        self.assertEqual(c.grad, 0.0)


class NetTest(unittest.TestCase):
    def test_neuron(self):
        n = N(3, True)
        w1 = V(.5)
        w2 = V(1.0)
        w3 = V(-1.0)
        n.weights = [w1, w2, w3]
        # relu(.5 * 4 + 1 * (-2) + (-1) * (-1) + 0.0) = 1
        out = n([V(4.0), V(-2.0), V(-1.0)])
        self.assertEqual(out.data, 1.0)
        out.backward()
        self.assertEqual(w1.grad, 4.0)
        self.assertEqual(w2.grad, -2.0)
        self.assertEqual(w3.grad, -1.0)


if __name__ == '__main__':
    unittest.main()
