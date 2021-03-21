import unittest

import numpy as np

from rt_opt import testproblems as tp
from rt_opt import testproblems_shifted as tps


def run_test(cls, testProb):
    if isinstance(testProb.min.x, tuple):
        if isinstance(testProb.min.f, tuple):
            for val in testProb.min.x:
                cls.assertGreater(testProb.f(val), testProb.min.f[0])
                cls.assertLess(testProb.f(val), testProb.min.f[1])
        else:
            for val in testProb.min.x:
                cls.assertAlmostEqual(testProb.f(val), testProb.min.f,
                                      delta=np.finfo(float).eps)
    else:
        if isinstance(testProb.min.f, tuple):
            cls.assertGreater(testProb.f(testProb.min.x), testProb.min.f[0])
            cls.assertLess(testProb.f(testProb.min.x), testProb.min.f[1])
        else:
            cls.assertAlmostEqual(testProb.f(testProb.min.x), testProb.min.f,
                                  delta=np.finfo(float).eps)


class Test_testproblems_2D(unittest.TestCase):
    def test_Ackley(self):
        testProb = tp.Ackley()
        run_test(self, testProb)

    def test_Beale(self):
        testProb = tp.Beale()
        run_test(self, testProb)

    def test_GoldsteinPrice(self):
        testProb = tp.GoldsteinPrice()
        run_test(self, testProb)

    def test_Booth(self):
        testProb = tp.Booth()
        run_test(self, testProb)

    def test_Bukin6(self):
        testProb = tp.Bukin6()
        run_test(self, testProb)

    def test_Matyas(self):
        testProb = tp.Matyas()
        run_test(self, testProb)

    def test_Levi13(self):
        testProb = tp.Levi13()
        run_test(self, testProb)

    def test_Himmelblau(self):
        testProb = tp.Himmelblau()
        run_test(self, testProb)

    def test_ThreeHumpCamel(self):
        testProb = tp.ThreeHumpCamel()
        run_test(self, testProb)

    def test_Easom(self):
        testProb = tp.Easom()
        run_test(self, testProb)

    def test_CrossInTray(self):
        testProb = tp.CrossInTray()
        run_test(self, testProb)

    def test_Eggholder(self):
        testProb = tp.Eggholder()
        run_test(self, testProb)

    def test_Hoelder(self):
        testProb = tp.Hoelder()
        run_test(self, testProb)

    def test_McCormick(self):
        testProb = tp.McCormick()
        run_test(self, testProb)

    def test_Schaffer2(self):
        testProb = tp.Schaffer2()
        run_test(self, testProb)

    def test_Schaffer4(self):
        testProb = tp.Schaffer4()
        run_test(self, testProb)


class Test_testproblems_shifted_2D(unittest.TestCase):
    def test_Ackley(self):
        testProb = tps.Ackley()
        run_test(self, testProb)

    def test_Beale(self):
        testProb = tps.Beale()
        run_test(self, testProb)

    def test_GoldsteinPrice(self):
        testProb = tps.GoldsteinPrice()
        run_test(self, testProb)

    def test_Booth(self):
        testProb = tps.Booth()
        run_test(self, testProb)

    def test_Bukin6(self):
        testProb = tps.Bukin6()
        run_test(self, testProb)

    def test_Matyas(self):
        testProb = tps.Matyas()
        run_test(self, testProb)

    def test_Levi13(self):
        testProb = tps.Levi13()
        run_test(self, testProb)

    def test_Himmelblau(self):
        testProb = tps.Himmelblau()
        run_test(self, testProb)

    def test_ThreeHumpCamel(self):
        testProb = tps.ThreeHumpCamel()
        run_test(self, testProb)

    def test_Easom(self):
        testProb = tps.Easom()
        run_test(self, testProb)

    def test_CrossInTray(self):
        testProb = tps.CrossInTray()
        run_test(self, testProb)

    def test_Eggholder(self):
        testProb = tps.Eggholder()
        run_test(self, testProb)

    def test_Hoelder(self):
        testProb = tps.Hoelder()
        run_test(self, testProb)

    def test_McCormick(self):
        testProb = tps.McCormick()
        run_test(self, testProb)

    def test_Schaffer2(self):
        testProb = tps.Schaffer2()
        run_test(self, testProb)

    def test_Schaffer4(self):
        testProb = tps.Schaffer4()
        run_test(self, testProb)


class Test_testproblems_nD(unittest.TestCase):
    def setUp(self):
        self.n_dims = 100

    def test_Rastrigin(self):
        testProb = tp.Rastrigin(self.n_dims)
        run_test(self, testProb)

    def test_Sphere(self):
        testProb = tp.Sphere(self.n_dims)
        run_test(self, testProb)

    def test_Rosenbrock(self):
        testProb = tp.Rosenbrock(self.n_dims)
        run_test(self, testProb)

    def test_StyblinskiTang(self):
        testProb = tp.StyblinskiTang(self.n_dims)
        run_test(self, testProb)


class Test_testproblems_shifted_nD(unittest.TestCase):
    def setUp(self):
        self.n_dims = 100

    def test_Rastrigin(self):
        testProb = tps.Rastrigin(self.n_dims)
        run_test(self, testProb)

    def test_Sphere(self):
        testProb = tps.Sphere(self.n_dims)
        run_test(self, testProb)

    def test_Rosenbrock(self):
        testProb = tps.Rosenbrock(self.n_dims)
        run_test(self, testProb)

    def test_StyblinskiTang(self):
        testProb = tps.StyblinskiTang(self.n_dims)
        run_test(self, testProb)


if __name__ == '__main__':
    unittest.main()
