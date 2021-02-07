import unittest

from rt_optimizer.testproblems import *


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
        testProb = Ackley()
        run_test(self, testProb)

    def test_Beale(self):
        testProb = Beale()
        run_test(self, testProb)

    def test_GoldsteinPrice(self):
        testProb = GoldsteinPrice()
        run_test(self, testProb)

    def test_Booth(self):
        testProb = Booth()
        run_test(self, testProb)

    def test_Bukin6(self):
        testProb = Bukin6()
        run_test(self, testProb)

    def test_Matyas(self):
        testProb = Matyas()
        run_test(self, testProb)

    def test_Levi13(self):
        testProb = Levi13()
        run_test(self, testProb)

    def test_Himmelblau(self):
        testProb = Himmelblau()
        run_test(self, testProb)

    def test_ThreeHumpCamel(self):
        testProb = ThreeHumpCamel()
        run_test(self, testProb)

    def test_Easom(self):
        testProb = Easom()
        run_test(self, testProb)

    def test_CrossInTray(self):
        testProb = CrossInTray()
        run_test(self, testProb)

    def test_Eggholder(self):
        testProb = Eggholder()
        run_test(self, testProb)

    def test_Hoelder(self):
        testProb = Hoelder()
        run_test(self, testProb)

    def test_McCormick(self):
        testProb = McCormick()
        run_test(self, testProb)

    def test_Schaffer2(self):
        testProb = Schaffer2()
        run_test(self, testProb)

    def test_Schaffer4(self):
        testProb = Schaffer4()
        run_test(self, testProb)


class Test_testproblems_nD(unittest.TestCase):
    def setUp(self):
        self.n_dims = 100

    def test_Rastrigin(self):
        testProb = Rastrigin(self.n_dims)
        run_test(self, testProb)

    def test_Sphere(self):
        testProb = Sphere(self.n_dims)
        run_test(self, testProb)

    def test_Rosenbrock(self):
        testProb = Rosenbrock(self.n_dims)
        run_test(self, testProb)

    def test_StyblinskiTang(self):
        testProb = StyblinskiTang(self.n_dims)
        run_test(self, testProb)


if __name__ == '__main__':
    unittest.main()
