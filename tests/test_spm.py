import unittest
from tsadmetrics.metrics.spm import *

from sklearn.metrics import fbeta_score
import numpy as np
import random


import unittest
import numpy as np

class TestDiceCoefficient(unittest.TestCase):

    def setUp(self):
        self.y_true  = np.array([0,0,0,0,0, 1,1,1,1,1,1,1,1, 0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1])

        self.y_pred1 = np.array([0,0,0,0,0, 1,1,1,1,1,1,1,1, 0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0])

        self.y_pred2 = np.array([0,0,0,0,0, 1,0,0,0,0,0,0,0, 0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0])

        self.y_pred3 = self.y_true.copy()

        self.y_pred4 = np.zeros(len(self.y_true), dtype=int)

    def test_scores(self):
        metric = DiceCoefficient()
        score = round(metric.compute(self.y_true, self.y_pred1), 2)
        expected = 0.67
        self.assertAlmostEqual(score, expected, places=4)

        score = round(metric.compute(self.y_true, self.y_pred2), 2)
        expected = 0.22
        self.assertAlmostEqual(score, expected, places=4)

        score = round(metric.compute(self.y_true, self.y_pred3), 2)
        expected = 1.00
        self.assertAlmostEqual(score, expected, places=4)

        score = round(metric.compute(self.y_true, self.y_pred4), 2)
        expected = 0.00
        self.assertAlmostEqual(score, expected, places=4)

    def test_all_zeros_returns_one(self):
        y_true = np.zeros(50, dtype=int)
        y_pred = np.zeros(50, dtype=int)
        metric = DiceCoefficient()
        score = metric.compute(y_true, y_pred)
        self.assertAlmostEqual(score, 1.0, places=6)

    def test_consistency(self):
        try:
            rng = np.random.default_rng(123)
            metric = DiceCoefficient()

            y_true = rng.integers(0, 2, size=100)
            y_pred = np.zeros(100, dtype=int)
            metric.compute(y_true, y_pred)

            for _ in range(1000):
                y_true = rng.integers(0, 2, size=200)
                y_pred = rng.integers(0, 2, size=200)
                _ = metric.compute(y_true, y_pred)
        except Exception as e:
            self.fail(f"DiceCoefficient raised an exception {e}")



class TestPointwiseFScore(unittest.TestCase):

    def setUp(self):
        """
        Configuración inicial para las pruebas.
        """
        self.num_tests = 100  
        self.test_cases = []
        for _ in range(self.num_tests):
            y_true = np.random.choice([0, 1], size=(10000,))
            y_pred = np.random.choice([0, 1], size=(10000,))
            self.test_cases.append((y_true, y_pred))

        y_true_perfect = np.random.choice([0, 1], size=(10000,))
        y_pred_perfect = y_true_perfect.copy()  
        self.test_cases.append((y_true_perfect, y_pred_perfect))
        
        y_true_all_zeros = np.random.choice([0, 1], size=(10000,))
        y_pred_all_zeros = np.zeros(10000, dtype=int)  
        self.test_cases.append((y_true_all_zeros, y_pred_all_zeros))
        


    
    def test(self):

        for y_true, y_pred in self.test_cases:
            with self.subTest(y_true=y_true, y_pred=y_pred):
                beta = random.randint(0,1000000)
                metric = PointwiseFScore(beta=beta)
                f_score = metric.compute(y_true, y_pred)
                expected_f_score = fbeta_score(y_true, y_pred, beta=beta)
                self.assertAlmostEqual(f_score, expected_f_score, places=4)

class TestPrecisionAtK(unittest.TestCase):

    def setUp(self):

 
        self.y_true1 =  np.array([0,0,1,1])


        self.y_pred1 = np.array([0.2, 0.9, 0.3, 0.8])

        self.y_pred2 = np.array([1, 2, 3, 4])

        self.y_pred3 = np.array([3, 4, 1, 2])

        self.y_true2 =  np.array([1,1,1,0])

        self.y_pred4 = np.array([3, 4, 1, 2])

        self.y_pred5 = self.y_true1
        self.y_pred6 = np.zeros(len(self.y_true1))


    

    def test(self):
        metric = PrecisionAtK()
        score = round(metric.compute(self.y_true1, self.y_pred1),2)
        expected_score = 0.5
        self.assertAlmostEqual(score, expected_score, places=4)

        score = round(metric.compute(self.y_true1, self.y_pred2),2)
        expected_score = 1
        self.assertAlmostEqual(score, expected_score, places=4)

        score = round(metric.compute(self.y_true1, self.y_pred3),2)
        expected_score = 0
        self.assertAlmostEqual(score, expected_score, places=4)

        score = round(metric.compute(self.y_true2, self.y_pred4),2)
        expected_score = round(2/3,2)
        self.assertAlmostEqual(score, expected_score, places=4)

        score = round(metric.compute(self.y_true1, self.y_pred5),2)
        expected_metric = 1.0
        self.assertAlmostEqual(score, expected_metric, places=4)

        score = round(metric.compute(self.y_true1, self.y_pred6),2)
        expected_metric = 0.5
        self.assertAlmostEqual(score, expected_metric, places=4)
        
    def test_consistency(self):
        try:
            metric = PrecisionAtK()
            for _ in range(100):
                y_true = np.random.choice([0, 1], size=(100,))
                y_pred = np.random.random( size=(100,))

                score = metric.compute(y_true, y_pred)
        except Exception as e:
            self.fail(f"PrecisionAtK raised an exception {e}")



class TestPointwiseAucRoc(unittest.TestCase):
    def setUp(self):
        """
        Configuración inicial para las pruebas.
        """
 
        self.y_true1 =  np.array([0,0,1,1])


        self.y_pred1 = np.array([1, 3, 2, 4])

        self.y_pred2 = np.array([1, 2, 3, 4])

        self.y_pred3 = np.array([4, 4, 4, 4])

        self.y_pred4 = self.y_true1
        self.y_pred5 = np.zeros(len(self.y_true1))
    

    def test(self):
        metric = PointwiseAucRoc()
        score = round(metric.compute(self.y_true1, self.y_pred1),2)
        expected_score = 0.75
        self.assertAlmostEqual(score, expected_score, places=4)

        score = round(metric.compute(self.y_true1, self.y_pred2),2)
        expected_score = 1
        self.assertAlmostEqual(score, expected_score, places=4)

        score = round(metric.compute(self.y_true1, self.y_pred3),2)
        expected_score = 0.5
        self.assertAlmostEqual(score, expected_score, places=4)

        score = round(metric.compute(self.y_true1, self.y_pred4),2)
        expected_metric = 1.0
        self.assertAlmostEqual(score, expected_metric, places=4)

        score = round(metric.compute(self.y_true1, self.y_pred5),2)
        expected_metric = 0.5
        self.assertAlmostEqual(score, expected_metric, places=4)

        
    def test_consistency(self):
        try:
            metric = PointwiseAucRoc()
            for _ in range(100):
                y_true = np.random.choice([0, 1], size=(100,))
                y_pred = np.random.random( size=(100,))

                score = metric.compute(y_true, y_pred)
        except Exception as e:
            self.fail(f"PointwiseAucRoc raised an exception {e}")


class TestPointwiseAucPr(unittest.TestCase):
    def setUp(self):
        """
        Configuración inicial para las pruebas.
        """
 
        self.y_true1 =  np.array([0,0,1,1])


        self.y_pred1 = np.array([1, 3, 2, 4])

        self.y_pred2 = np.array([1, 2, 3, 4])

        self.y_pred3 = np.array([4, 4, 4, 4])

        self.y_pred4 = self.y_true1
        self.y_pred5 = np.zeros(len(self.y_true1))
    

    def test(self):
        """
        Prueba para la función metric.compute.
        """
        metric = PointwiseAucPr()
        score = round(metric.compute(self.y_true1, self.y_pred1),2)
        expected_score = 0.83
        self.assertAlmostEqual(score, expected_score, places=4)

        score = round(metric.compute(self.y_true1, self.y_pred2),2)
        expected_score = 1
        self.assertAlmostEqual(score, expected_score, places=4)

        score = round(metric.compute(self.y_true1, self.y_pred3),2)
        expected_score = 0.5
        self.assertAlmostEqual(score, expected_score, places=4)

        score = round(metric.compute(self.y_true1, self.y_pred4),2)
        expected_metric = 1.0
        self.assertAlmostEqual(score, expected_metric, places=4)

        score = round(metric.compute(self.y_true1, self.y_pred5),2)
        expected_metric = 0.5
        self.assertAlmostEqual(score, expected_metric, places=4)

        
    def test_consistency(self):
        try:
            metric = PointwiseAucPr()
            for _ in range(100):
                y_true = np.random.choice([0, 1], size=(100,))
                y_pred = np.random.random( size=(100,))

                score = metric.compute(y_true, y_pred)
        except Exception as e:
            self.fail(f"auc_pr raised an exception {e}")