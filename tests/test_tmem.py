import unittest

import numpy as np
import random

import unittest
import numpy as np
from tsadmetrics.metrics.tem.tmem import *

class TestTemporalDistance(unittest.TestCase):

    def setUp(self):
        self.y_true1 = np.array([0, 0, 1, 1, 0, 0])
        self.y_pred1 = np.array([0, 0, 1, 1, 0, 0])

        self.y_true2 = np.array([0, 0, 0, 1, 1, 0])
        self.y_pred2 = np.array([0, 1, 1, 0, 0, 0])

        self.y_true3 = np.array([0, 0, 1, 1, 0, 0])
        self.y_pred3 = np.array([1, 1, 0, 0, 0, 0])

        self.y_pred4 = self.y_true1
        self.y_pred5 = np.zeros(len(self.y_true1))

    def test_temporal_distance_euclidean(self):
        metric = TemporalDistance(distance=0)

        td = metric.compute(self.y_true1, self.y_pred1)
        expected = 0
        self.assertEqual(td, expected)

        td = metric.compute(self.y_true2, self.y_pred2)
        expected = 6
        self.assertEqual(td, expected)

        td = metric.compute(self.y_true3, self.y_pred3)
        expected = 6
        self.assertEqual(td, expected)

        score = round(metric.compute(self.y_true1, self.y_pred4),2)
        expected_metric = 0
        self.assertAlmostEqual(score, expected_metric, places=4)

        score = round(metric.compute(self.y_true1, self.y_pred5),2)
        expected_metric = 12
        self.assertAlmostEqual(score, expected_metric, places=4)

    def test_temporal_distance_squared(self):
        metric = TemporalDistance(distance=1)

        td = metric.compute(self.y_true1, self.y_pred1)
        expected = 0
        self.assertEqual(td, expected)

        td = metric.compute(self.y_true2, self.y_pred2)
        expected = 18
        self.assertEqual(td, expected)

        td = metric.compute(self.y_true3, self.y_pred3)
        expected = 18
        self.assertEqual(td, expected)

        score = round(metric.compute(self.y_true1, self.y_pred4),2)
        expected_metric = 0
        self.assertAlmostEqual(score, expected_metric, places=4)

        score = round(metric.compute(self.y_true1, self.y_pred5),2)
        expected_metric = 144
        self.assertAlmostEqual(score, expected_metric, places=4)

    def test_consistency(self):
        try:
            
            for _ in range(100):
                y_true = np.random.choice([0, 1], size=(100,))
                y_pred = np.zeros(100)
                metric=TemporalDistance(distance=random.choice([0, 1]))
                metric.compute(y_true, y_pred)
        except Exception as e:
            self.fail(f"absolute_detection_distance raised an exception {e}")


class TestAbsoluteDetectionDistance(unittest.TestCase):

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
        metric = AbsoluteDetectionDistance()
        score = round(metric.compute(self.y_true1, self.y_pred1),2)
        expected_score = 0.25
        self.assertAlmostEqual(score, expected_score, places=4)

        score = round(metric.compute(self.y_true1, self.y_pred2),2)
        expected_score = 0.25
        self.assertAlmostEqual(score, expected_score, places=4)

        score = round(metric.compute(self.y_true2, self.y_pred21),2)
        expected_score = 0.06
        self.assertAlmostEqual(score, expected_score, places=4)

        score = round(metric.compute(self.y_true2, self.y_pred22),2)
        expected_score = 0.12
        self.assertAlmostEqual(score, expected_score, places=4)

        score = round(metric.compute(self.y_true1, self.y_pred3),2)
        expected_metric = 0.17 #The mean of the distances is never 0
        self.assertAlmostEqual(score, expected_metric, places=4)

        score = round(metric.compute(self.y_true1, self.y_pred4),2)
        expected_metric = 0 
        self.assertAlmostEqual(score, expected_metric, places=4)

        
    def testconsistency(self):
        try:
            y_true = np.random.choice([0, 1], size=(100,))
            y_pred = np.zeros(100)
            metric = AbsoluteDetectionDistance()
            for _ in range(100):
                y_true = np.random.choice([0, 1], size=(100,))
                y_pred = np.random.choice([0, 1], size=(100,))

                score = metric.compute(y_true, y_pred)
        except Exception as e:
            self.fail(f"AbsoluteDetectionDistance raised an exception {e}")


class TestEnhancedTimeseriesAwareFScore(unittest.TestCase):

    def setUp(self):

        self.y_true1 = np.array([0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1])
        self.y_pred1 = np.array([0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        self.y_pred2 = np.array([0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0])

        self.y_true2 = np.array([0,0,1,0,1,0,1,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0])
        self.y_pred21 = np.array([0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        self.y_pred22 = np.array([0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0])

        self.y_pred3 = self.y_true1
        self.y_pred4 = np.zeros(len(self.y_true1))

    

    def test(self):
        metric = EnhancedTimeseriesAwareFScore(theta_p=0.5, theta_r=0.1)
        f_score = round(metric.compute(self.y_true1, self.y_pred1),2)
        expected_f_score = 0.67
        self.assertAlmostEqual(f_score, expected_f_score, places=4)

        f_score = round(metric.compute(self.y_true1, self.y_pred2),2)
        expected_f_score = 0.72
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
        try:
            y_true = np.random.choice([0, 1], size=(100,))
            y_pred = np.zeros(100)
            metric = EnhancedTimeseriesAwareFScore(theta_r=random.random(), theta_p=random.random())
            metric.compute(y_true, y_pred)
            for _ in range(100):
                y_true = np.random.choice([0, 1], size=(100,))
                y_pred = np.random.choice([0, 1], size=(100,))

                f_score = metric.compute(y_true, y_pred)
        except Exception as e:
            self.fail(f"EnhancedTimeseriesAwareFScore raised an exception {e}")


