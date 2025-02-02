import unittest
from partie_1 import rp, err_add, err_mul

class TestPartie1(unittest.TestCase):

    def assert_eg(self, a, b):
        self.assertEqual(a, b, f"KO: valeur_reel={a} et rp={b}")

    def assert_in(self, a, b):
        self.assertLessEqual(a, b, f"KO: err_trouver:{a} et precision:{b} et l'ecart entre les deux est {abs(a-b)}")

    def test_rp(self):
        self.assert_eg(3.142, rp(3.141592658, 4))
        self.assert_eg(3.14159, rp(3.141592658, 6))
        self.assert_eg(10510, rp(10507.1823, 4))
        self.assert_eg(10507.2, rp(10507.1823, 6))
        self.assert_eg(0.0001858, rp(0.0001857563, 4))
        self.assert_eg(0.000185756, rp(0.0001857563, 6))

    def test_simule_err_add(self):
        a = 3.141592658
        b = 4.0001857563
        c = 6.1254879633154
        self.assert_in(err_add(a, b, 6), 10**(-5))
        self.assert_in(err_add(a, c, 6), 10**(-5))
        self.assert_in(err_add(b, c, 6), 10**(-5))

    def test_simule_err_mul(self):
        a = 3.141592658
        b = 4.0001857563
        c = 6.1254879633154
        self.assert_in(err_mul(a, b, 6), 10**(-5))
        self.assert_in(err_mul(a, c, 6), 10**(-5))
        self.assert_in(err_mul(b, c, 6), 10**(-5))

if __name__ == '__main__':
    unittest.main()