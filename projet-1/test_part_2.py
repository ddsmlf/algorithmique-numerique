import unittest
import math
import logarithme_neperien_de_2
import cordic

class TestCordicFunctions(unittest.TestCase):

    def test_log_2_approximation(self):
        p = 4 # Example precision and because starting from 5 it takes a lot of time the while
        approximation, error = logarithme_neperien_de_2.log_2_approximation(p)
        self.assertAlmostEqual(approximation, round(math.log(2), p))
        self.assertAlmostEqual(error, abs((math.log(2) - round(math.log(2), p)) / math.log(2)))

    def test_fonction_err(self):
        self.assertEqual(cordic.err(0.003491830023001, 0.003491830053001), 8)
        self.assertEqual(cordic.err(133233.1231234993, 133233.1231234993), 16)
        self.assertEqual(cordic.err(3.14159265358, 3.14159265359), 11)

    def test_cordic_exp(self):
        print("")
        print("")
        print("test exp")
        precision = 12
        A, B = [], []
        for i in range(1, 301):
            err = cordic.err(cordic.cordic_exp(i), math.exp(i))
            if precision > err:
                if err not in A:
                    A.append(err)
                    B.append(1)
                else:
                    j = A.index(err)
                    B[j] += 1
        somme = 300 - sum(B)
        print("pour les entiers de 1 à 300:")   
        print(*(f"    {B[i]*100/300}% ({B[i]}) valeur possede une precision {A[i]}" for i in range(len(A))), sep="\n")
        print(f"    {somme*100/300}% ({somme}) valeur possede une precision superieur ou egal à 12")

    def test_cordic_ln(self):
        print("")
        print("")
        print("test ln")
        precision = 12
        A, B = [], []
        for i in range(2, 301):
            err = cordic.err(cordic.cordic_ln(i), math.log(i))
            if precision > err:
                if err not in A:
                    A.append(err)
                    B.append(1)
                else:
                    j = A.index(err)
                    B[j] += 1
        somme = 300 - sum(B)
        print("pour les entiers de 1 à 300:")
        print(*(f"    {B[i]*100/300}% ({B[i]}) valeur possede une precision {A[i]}" for i in range(len(A))), sep="\n")
        print(f"    {somme*100/300}% ({somme}) valeur possede une precision superieur ou egal à 12")

    def test_cordic_tan(self):
        print("")
        print("")
        print("test tan")
        precision = 12
        A, B = [], []
        for i in range(1, 301):
            err = cordic.err(cordic.cordic_tan(i), math.tan(i))
            if precision > err:
                if err not in A:
                    A.append(err)
                    B.append(1)
                else:
                    j = A.index(err)
                    B[j] += 1
        somme = 300 - sum(B)
        print("pour les entiers de 1 à 300:")
        print(*(f"    {B[i]*100/300}% ({B[i]}) valeur possede une precision {A[i]}" for i in range(len(A))), sep="\n")
        print(f"    {somme*100/300}% ({somme}) valeur possede une precision superieur ou egal à 12")

    def test_cordic_atan(self):
        print("")
        print("")
        print("test atan")
        precision = 12
        A, B = [], []
        for i in range(1, 301):
            err = cordic.err(cordic.cordic_arctan(i), math.atan(i))
            if precision > err:
                if err not in A:
                    A.append(err)
                    B.append(1)
                else:
                    j = A.index(err)
                    B[j] += 1
        somme = 300 - sum(B)
        print("pour les entiers de 1 à 300:")
        print(*(f"    {B[i]*100/300}% ({B[i]}) valeur possede une precision {A[i]}" for i in range(len(A))), sep="\n")
        print(f"    {somme*100/300}% ({somme}) valeur possede une precision superieur ou egal à 12")

    def test_float(self):
        a = 3.141592653589793
        b = 0.001001645450
        c = 1.6834

        self._test_float_function(cordic.cordic_ln, math.log, a, b, c)
        self._test_float_function(cordic.cordic_exp, math.exp, a, b, c)
        self._test_float_function(cordic.cordic_arctan, math.atan, a, b, c)
        self._test_float_function(cordic.cordic_tan, math.tan, a, b, c)

    def _test_float_function(self, cordic_func, math_func, *values):
        for value in values:
            err = cordic.err(math_func(value), cordic_func(value))
            if err >= 12:
                self.assertTrue(True)
            else:
                self.assertTrue(False)

if __name__ == "__main__":
    unittest.main()
