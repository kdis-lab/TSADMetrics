import unittest
from tsadmetrics.metrics.tem.ptdm import *

import numpy as np
import random

class TestAverageDetectionCount(unittest.TestCase):


    def setUp(self):
        """
        Configuración inicial para las pruebas.
        """
        self.y_true1 = np.array([0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1])
        self.y_pred1 = np.array([0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        self.y_pred2 = np.array([0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0])

        self.y_true2  = np.array([0,0,0,1,1,0,0,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1])
        self.y_pred21 = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1])
        self.y_pred22 = np.array([0,0,0,1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0])

        self.y_pred3 = self.y_true1
        self.y_pred4 = np.zeros(len(self.y_true1))

    

    def test(self):
        metric = AverageDetectionCount()
        score = round(metric.compute(self.y_true1, self.y_pred1),2)
        expected_metric = 0.5
        self.assertAlmostEqual(score, expected_metric, places=4)

        score = round(metric.compute(self.y_true1, self.y_pred2),2)
        expected_metric = 0.12
        self.assertAlmostEqual(score, expected_metric, places=4)

        score = round(metric.compute(self.y_true2, self.y_pred21),2)
        expected_metric = 0.33
        self.assertAlmostEqual(score, expected_metric, places=4)

        score = round(metric.compute(self.y_true2, self.y_pred22),2)
        expected_metric = 0.67
        self.assertAlmostEqual(score, expected_metric, places=4)

        score = round(metric.compute(self.y_true1, self.y_pred3),2)
        expected_metric = 1.0
        self.assertAlmostEqual(score, expected_metric, places=4)

        score = round(metric.compute(self.y_true1, self.y_pred4),2)
        expected_metric = 0
        self.assertAlmostEqual(score, expected_metric, places=4)

        
    def test_consistency(self):
        metric = AverageDetectionCount()
        try:
            y_true = np.random.choice([0, 1], size=(100,))
            y_pred = np.zeros(100)
            metric.compute(y_true, y_pred)
            for _ in range(100):
                y_true = np.random.choice([0, 1], size=(100,))
                y_pred = np.random.choice([0, 1], size=(100,))

                score = metric.compute(y_true, y_pred)
        except Exception as e:
            self.fail(f"AverageDetectionCount raised an exception {e}")


class TestDetectionAccuracyInRange(unittest.TestCase):

    def setUp(self):
        """
        Configuración inicial para las pruebas.
        """
        self.y_true1 = np.array([0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1])
        self.y_pred1 = np.array([0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        self.y_pred2 = np.array([0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0])

        self.y_true2  = np.array([0,0,0,1,1,0,0,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1])
        self.y_pred21 = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1])
        self.y_pred22 = np.array([0,0,0,1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0])

        self.y_pred3 = self.y_true1
        self.y_pred4 = np.zeros(len(self.y_true1))

    def test(self):
        metric = DetectionAccuracyInRange(k=3)
        score = round(metric.compute(self.y_true1, self.y_pred1),2)
        expected_score = 1.0
        self.assertAlmostEqual(score, expected_score, places=4)

        score = round(metric.compute(self.y_true1, self.y_pred2),2)
        expected_score = 1.0
        self.assertAlmostEqual(score, expected_score, places=4)

        score = round(metric.compute(self.y_true2, self.y_pred21),2)
        expected_score = 1.0
        self.assertAlmostEqual(score, expected_score, places=4)

        score = round(metric.compute(self.y_true2, self.y_pred22),2)
        expected_score = 1.0
        self.assertAlmostEqual(score, expected_score, places=4)

        score = round(metric.compute(self.y_true1, self.y_pred3),2)
        expected_metric = 1.0
        self.assertAlmostEqual(score, expected_metric, places=4)

        score = round(metric.compute(self.y_true1, self.y_pred4),2)
        expected_metric = 0
        self.assertAlmostEqual(score, expected_metric, places=4)

    

        
    def test_consistency(self):
        metric = DetectionAccuracyInRange(k=4)
        try:
            y_true = np.random.choice([0, 1], size=(100,))
            y_pred = np.zeros(100)
            
            for _ in range(100):
                y_true = np.random.choice([0, 1], size=(100,))
                y_pred = np.random.choice([0, 1], size=(100,))

                score = metric.compute(y_true, y_pred)
        except Exception as e:
            self.fail(f"DetectionAccuracyInRange raised an exception {e}")






class TestPointadjustedAtKFScore(unittest.TestCase):

    def setUp(self):
        self.y_true  = np.array([0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1])
        self.y_pred1 = np.array([0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        self.y_pred2 = np.array([0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0])

        self.y_pred3 = self.y_true
        self.y_pred4 = np.zeros(len(self.y_true))


    def test(self):
        metric = PointadjustedAtKFScore(k=0.2)
        f_score = round(metric.compute(self.y_true, self.y_pred1),2)
        expected_f_score = 0.67
        self.assertAlmostEqual(f_score, expected_f_score, places=4)

        f_score = round(metric.compute(self.y_true, self.y_pred2),2)
        expected_f_score = 0.22
        self.assertAlmostEqual(f_score, expected_f_score, places=4)

        score = round(metric.compute(self.y_true, self.y_pred3),2)
        expected_metric = 1.0
        self.assertAlmostEqual(score, expected_metric, places=4)

        score = round(metric.compute(self.y_true, self.y_pred4),2)
        expected_metric = 0
        self.assertAlmostEqual(score, expected_metric, places=4)
        
    def test_consistency(self):
        metric = PointadjustedAtKFScore(k=0.3)
        try:
            y_true = np.random.choice([0, 1], size=(100,))
            y_pred = np.zeros(100)
            metric.compute(y_true, y_pred)
            for _ in range(1000):
                y_true = np.random.choice([0, 1], size=(100,))
                y_pred = np.random.choice([0, 1], size=(100,))
                f_score = metric.compute(y_true, y_pred)
        except Exception as e:
            self.fail(f"PointadjustedAtKFScore raised an exception {e}")

class TestPointadjustedAtKLFScore(unittest.TestCase):

    def setUp(self):
        self.y_true  = np.array([0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1])
        self.y_pred1 = np.array([0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        self.y_pred2 = np.array([0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0])

        self.y_pred3 = self.y_true
        self.y_pred4 = np.zeros(len(self.y_true))


    def test(self):
        metric = PointadjustedAtKLFScore(k=0.2,l=1000)
        f_score = round(metric.compute(self.y_true, self.y_pred1),2)
        expected_f_score = 0.67
        self.assertAlmostEqual(f_score, expected_f_score, places=4)

        metric = PointadjustedAtKLFScore(k=0.2,l=2)
        f_score = round(metric.compute(self.y_true, self.y_pred2),2)
        expected_f_score = 0.61
        self.assertAlmostEqual(f_score, expected_f_score, places=4)

        score = round(metric.compute(self.y_true, self.y_pred3),2)
        expected_metric = 1.0
        self.assertAlmostEqual(score, expected_metric, places=4)

        score = round(metric.compute(self.y_true, self.y_pred4),2)
        expected_metric = 0
        self.assertAlmostEqual(score, expected_metric, places=4)
        
    def test_consistency(self):
        metric = PointadjustedAtKLFScore(k=0.3,l=2)
        try:
            y_true = np.random.choice([0, 1], size=(100,))
            y_pred = np.zeros(100)
            metric.compute(y_true, y_pred)
            for _ in range(1000):
                y_true = np.random.choice([0, 1], size=(100,))
                y_pred = np.random.choice([0, 1], size=(100,))
                f_score = metric.compute(y_true, y_pred)
        except Exception as e:
            self.fail(f"PointadjustedAtKFScore raised an exception {e}")

class TestTimeseriesAwareFScore(unittest.TestCase):

    def setUp(self):
        """
        Configuración inicial para las pruebas.
        """
        self.y_true1 = np.array([0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1])
        self.y_pred1 = np.array([0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        self.y_pred2 = np.array([0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0])

        self.y_true2 = np.array([0,0,1,0,1,0,1,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0])
        self.y_pred21 = np.array([0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        self.y_pred22 = np.array([0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0])

        self.y_pred3 = self.y_true1
        self.y_pred4 = np.zeros(len(self.y_true1))

    

    def test(self):

        metric = TimeseriesAwareFScore(beta=1, alpha=0.5,delta=0, theta=0.5)
        f_score = round(metric.compute(self.y_true1, self.y_pred1),2)
        expected_f_score = 0.67
        self.assertAlmostEqual(f_score, expected_f_score, places=4)

        f_score = round(metric.compute(self.y_true1, self.y_pred2),2)
        expected_f_score = 0.12
        self.assertAlmostEqual(f_score, expected_f_score, places=4)

        f_score = round(metric.compute(self.y_true2, self.y_pred21),2)
        expected_f_score = 0.77
        self.assertAlmostEqual(f_score, expected_f_score, places=4)

        f_score = round(metric.compute(self.y_true2, self.y_pred22),2)
        expected_f_score = 0.67
        self.assertAlmostEqual(f_score, expected_f_score, places=4)

        score = round(metric.compute(self.y_true1, self.y_pred3),2)
        expected_metric = 1.0
        self.assertAlmostEqual(score, expected_metric, places=4)

        score = round(metric.compute(self.y_true1, self.y_pred4),2)
        expected_metric = 0
        self.assertAlmostEqual(score, expected_metric, places=4)
        
    def test_consistency(self):
        metric = TimeseriesAwareFScore(beta=1, alpha=0.5,delta=0, theta=0.5)
        try:
            y_true = np.random.choice([0, 1], size=(100,))
            y_pred = np.zeros(100)
            metric.compute(y_true, y_pred)
            for _ in range(100):
                y_true = np.random.choice([0, 1], size=(100,))
                y_pred = np.random.choice([0, 1], size=(100,))
                metric = TimeseriesAwareFScore(beta=1, alpha=random.random(),delta=0, theta=random.random())
                f_score = metric.compute(y_true, y_pred)
        except Exception as e:
            self.fail(f"ts_aware_f_score raised an exception {e}")




class TestTotalDetectedInRange(unittest.TestCase):

    def setUp(self):
        """
        Configuración inicial para las pruebas.
        """
        self.y_true1 = np.array([0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1])
        self.y_pred1 = np.array([0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        self.y_pred2 = np.array([0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0])

        self.y_true2  = np.array([0,0,0,1,1,0,0,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1])
        self.y_pred21 = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1])
        self.y_pred22 = np.array([0,0,0,1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0])

        self.y_pred3 = self.y_true1
        self.y_pred4 = np.zeros(len(self.y_true1))

    def test(self):
        metric = TotalDetectedInRange(k=3)
        score = round(metric.compute(self.y_true1, self.y_pred1),2)
        expected_score = 0.5
        self.assertAlmostEqual(score, expected_score, places=4)

        score = round(metric.compute(self.y_true1, self.y_pred2),2)
        expected_score = 0.5
        self.assertAlmostEqual(score, expected_score, places=4)

        score = round(metric.compute(self.y_true2, self.y_pred21),2)
        expected_score = 0.56
        self.assertAlmostEqual(score, expected_score, places=4)

        score = round(metric.compute(self.y_true2, self.y_pred22),2)
        expected_score = 0.44
        self.assertAlmostEqual(score, expected_score, places=4)

        score = round(metric.compute(self.y_true1, self.y_pred3),2)
        expected_metric = 1.0
        self.assertAlmostEqual(score, expected_metric, places=4)

        score = round(metric.compute(self.y_true1, self.y_pred4),2)
        expected_metric = 0
        self.assertAlmostEqual(score, expected_metric, places=4)

    

        
    def test_consistency(self):
        metric = TotalDetectedInRange(k=4)
        try:
            y_true = np.random.choice([0, 1], size=(100,))
            y_pred = np.zeros(100)
            metric.compute(y_true, y_pred)
            for _ in range(100):
                y_true = np.random.choice([0, 1], size=(100,))
                y_pred = np.random.choice([0, 1], size=(100,))

                score = metric.compute(y_true, y_pred)
        except Exception as e:
            self.fail(f"TotalDetectedInRange raised an exception {e}")





class TestWeightedDetectionDifference(unittest.TestCase):

    def setUp(self):
        """
        Configuración inicial para las pruebas.
        """
        self.y_true1 = np.array([0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1])
        self.y_pred1 = np.array([0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        self.y_pred2 = np.array([0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0])

        self.y_true2  = np.array([0,0,0,1,1,0,0,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1])
        self.y_pred21 = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1])
        self.y_pred22 = np.array([0,0,0,1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0])

        self.y_pred3 = self.y_true1
        self.y_pred4 = np.zeros(len(self.y_true1))

    def test(self):
        metric = WeightedDetectionDifference(k=3)
        score = round(metric.compute(self.y_true1, self.y_pred1),2)
        expected_score = 18.89
        self.assertAlmostEqual(score, expected_score, places=4)

        score = round(metric.compute(self.y_true1, self.y_pred2),2)
        expected_score = 24.89
        self.assertAlmostEqual(score, expected_score, places=4)

        score = round(metric.compute(self.y_true2, self.y_pred21),2)
        expected_score = 15.73
        self.assertAlmostEqual(score, expected_score, places=4)

        score = round(metric.compute(self.y_true2, self.y_pred22),2)
        expected_score = 16.73
        self.assertAlmostEqual(score, expected_score, places=4)

        score = round(metric.compute(self.y_true1, self.y_pred3),2)
        expected_metric = 10
        self.assertGreater(score, expected_metric)

        score = round(metric.compute(self.y_true1, self.y_pred4),2)
        expected_metric = 0
        self.assertAlmostEqual(score, expected_metric, places=4)

    

        
    def test_consistency(self):
        metric = WeightedDetectionDifference(k=4)
        try:
            y_true = np.random.choice([0, 1], size=(100,))
            y_pred = np.zeros(100)
            metric.compute(y_true, y_pred)
            for _ in range(100):
                y_true = np.random.choice([0, 1], size=(100,))
                y_pred = np.random.choice([0, 1], size=(100,))

                score = metric.compute(y_true, y_pred)
        except Exception as e:
            self.fail(f"WeightedDetectionDifference raised an exception {e}")
