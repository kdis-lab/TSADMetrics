import unittest

import numpy as np
import random

import unittest
import numpy as np
from tsadmetrics.metrics.tem.dpm import *

class TestDelayThresholdedPointadjustedFScore(unittest.TestCase):

    def setUp(self):

        self.y_true  = np.array([0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1])
        self.y_pred1 = np.array([0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        self.y_pred2 = np.array([0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0])
        self.y_pred3 = self.y_true
        self.y_pred4 = np.zeros(len(self.y_true))

    

    def test(self):
        metric = DelayThresholdedPointadjustedFScore(k=2, beta=1.0)
        f_score = round(metric.compute(self.y_true, self.y_pred1),2)
        expected_f_score =0.67
        self.assertAlmostEqual(f_score, expected_f_score, places=4)
        
        f_score = round(metric.compute(self.y_true, self.y_pred2),2)
        expected_f_score =1
        self.assertAlmostEqual(f_score, expected_f_score, places=4)

        score = round(metric.compute(self.y_true, self.y_pred3),2)
        expected_metric =1.0
        self.assertAlmostEqual(score, expected_metric, places=4)

        score = round(metric.compute(self.y_true, self.y_pred4),2)
        expected_metric =0
        self.assertAlmostEqual(score, expected_metric, places=4)

    def test_consistency(self):
        try:
            y_true = np.random.choice([0,1], size=(100,))
            y_pred = np.zeros(100)
            metric = DelayThresholdedPointadjustedFScore(k=2, beta=1.0)
            metric.compute(y_true, y_pred)
            for _ in range(1000):
                y_true = np.random.choice([0,1], size=(100,))
                y_pred = np.random.choice([0,1], size=(100,))
                f_score = metric.compute(y_true, y_pred)
        except Exception as e:
            self.fail(f"DelayThresholdedPointadjustedFScore raised an exception {e}")


class TestEarlyDetectionScore(unittest.TestCase):

    def setUp(self):
        self.y_true =  np.array([0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1])

        self.y_pred1 = np.array([0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1])

        self.y_pred2 = np.array([0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0])
        self.y_pred3 = np.zeros(len(self.y_true))

        self.y_pred4 = np.array([0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0])

    def test_scores(self):
        metric = EarlyDetectionScore()

        score1 = round(metric.compute(self.y_true, self.y_pred1), 2)
        expected1 =1.0  
        self.assertAlmostEqual(score1, expected1, places=4)

        score2 = round(metric.compute(self.y_true, self.y_pred2), 2)
        expected2 =0.0  
        self.assertAlmostEqual(score2, expected2, places=4)

        score3 = round(metric.compute(self.y_true, self.y_pred3), 2)
        expected3 =0.0  
        self.assertAlmostEqual(score3, expected3, places=4)

        score4 = round(metric.compute(self.y_true, self.y_pred4), 2)
        expected4 =1 
        self.assertAlmostEqual(score4, expected4, places=4)

    def test_consistency(self):
        try:
            y_true = np.random.choice([0,1], size=(100,))
            y_pred = np.zeros(100)
            metric = EarlyDetectionScore()
            metric.compute(y_true, y_pred)

            for _ in range(1000):
                y_true = np.random.choice([0,1], size=(100,))
                y_pred = np.random.choice([0,1], size=(100,))
                metric.compute(y_true, y_pred)
        except Exception as e:
            self.fail(f"EarlyDetectionScore raised an exception {e}")


class TestLatencySparsityawareFScore(unittest.TestCase):

    def setUp(self):

        self.y_true  = np.array([0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1])
        self.y_pred1 = np.array([0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        self.y_pred2 = np.array([0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0])
        self.y_pred3 = self.y_true
        self.y_pred4 = np.zeros(len(self.y_true))

    def test(self):
        metric = LatencySparsityawareFScore(ni=2, beta=1.0)
        f_score = round(metric.compute(self.y_true, self.y_pred1),2)
        expected_f_score =0.71
        self.assertAlmostEqual(f_score, expected_f_score, places=4)

        f_score = round(metric.compute(self.y_true, self.y_pred2),2)
        expected_f_score =1
        self.assertAlmostEqual(f_score, expected_f_score, places=4)

        score = round(metric.compute(self.y_true, self.y_pred3),2)
        expected_metric =1.0
        self.assertAlmostEqual(score, expected_metric, places=4)

        score = round(metric.compute(self.y_true, self.y_pred4),2)
        expected_metric =0
        self.assertAlmostEqual(score, expected_metric, places=4)
        
    def test_consistency(self):
        try:
            y_true = np.random.choice([0,1], size=(100,))
            y_pred = np.zeros(100)
            metric = LatencySparsityawareFScore(ni=2, beta=1.0)
            metric.compute(y_true, y_pred)
            for _ in range(1000):
                y_true = np.random.choice([0,1], size=(100,))
                y_pred = np.random.choice([0,1], size=(100,))
                f_score = metric.compute(y_true, y_pred)
        except Exception as e:
            self.fail(f"LatencySparsityawareFScore raised an exception {e}")


class TestMeanTimeToDetect(unittest.TestCase):

    def setUp(self):

        self.y_true1 = np.array([0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1])
        self.y_pred1 = np.array([0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        self.y_pred2 = np.array([0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0])

        self.y_true2  = np.array([0,0,0,1,1,0,0,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1])
        self.y_pred21 = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1])
        self.y_pred22 = np.array([0,0,0,1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0])

        self.y_pred3 = self.y_true1
        self.y_pred4 = np.zeros(len(self.y_true1))

    def test(self):
        metric = MeanTimeToDetect()
        score = round(metric.compute(self.y_true1, self.y_pred1),2)
        expected_score =0.0
        self.assertAlmostEqual(score, expected_score, places=4)

        score = round(metric.compute(self.y_true1, self.y_pred2),2)
        expected_score =0.0
        self.assertAlmostEqual(score, expected_score, places=4)

        score = round(metric.compute(self.y_true2, self.y_pred21),2)
        expected_score = 8.0
        self.assertAlmostEqual(score, expected_score, places=4)

        score = round(metric.compute(self.y_true2, self.y_pred22),2)
        expected_score =0.0
        self.assertAlmostEqual(score, expected_score, places=4)

        score = round(metric.compute(self.y_true1, self.y_pred3),2)
        expected_metric =0.0
        self.assertAlmostEqual(score, expected_metric, places=4)

        score = round(metric.compute(self.y_true1, self.y_pred4),2)
        expected_metric =0.0
        self.assertAlmostEqual(score, expected_metric, places=4)


    

        
    def test_consistency(self):
        try:

            y_true = np.random.choice([0,1], size=(100,))
            y_pred = np.zeros(100)
            metric = MeanTimeToDetect()
            metric.compute(y_true, y_pred)
            for _ in range(100):
                y_true = np.random.choice([0,1], size=(100,))
                y_pred = np.random.choice([0,1], size=(100,))

                score = metric.compute(y_true, y_pred)
        except Exception as e:
            self.fail(f"MeanTimeToDetect raised an exception {e}")


class TestNabScore(unittest.TestCase):

    def setUp(self):

        self.y_true1 = np.array([0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1])
        self.y_pred1 = np.array([0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        self.y_pred2 = np.array([0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0])

        self.y_true2  = np.array([0,0,0,1,1,0,0,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1])
        self.y_pred21 = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1])
        self.y_pred22 = np.array([0,0,0,1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0])

        self.y_pred3 = self.y_true1
        self.y_pred4 = np.zeros(len(self.y_true1))

    

    def test(self):
        metric = NabScore()
        f_score = round(metric.compute(self.y_true1, self.y_pred1),2)
        expected_f_score = 50
        self.assertAlmostEqual(f_score, expected_f_score, places=4)

        f_score = round(metric.compute(self.y_true1, self.y_pred2),2)
        expected_f_score =100
        self.assertAlmostEqual(f_score, expected_f_score, places=4)

        f_score = round(metric.compute(self.y_true2, self.y_pred21),2)
        expected_f_score = 33.33
        self.assertAlmostEqual(f_score, expected_f_score, places=4)

        f_score = round(metric.compute(self.y_true2, self.y_pred22),2)
        expected_f_score = 66.67
        self.assertAlmostEqual(f_score, expected_f_score, places=4)

        score = round(metric.compute(self.y_true1, self.y_pred3),2)
        expected_metric =100
        self.assertAlmostEqual(score, expected_metric, places=4)

        score = round(metric.compute(self.y_true1, self.y_pred4),2)
        expected_metric =0
        self.assertAlmostEqual(score, expected_metric, places=4)
        
    def test_consistency(self):
        try:
            metric = NabScore()
            y_true = np.random.choice([0,1], size=(100,))
            y_pred = np.zeros(100)
            metric.compute(y_true, y_pred)
            for _ in range(100):
                y_true = np.random.choice([0,1], size=(100,))
                y_pred = np.random.choice([0,1], size=(100,))

                score = metric.compute(y_true, y_pred)
        except Exception as e:
            self.fail(f"NabScore raised an exception {e}")