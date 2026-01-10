import unittest
from tsadmetrics.metrics.tem.tpdm import *

import numpy as np
import random


class TestBalancedPointadjustedFScore(unittest.TestCase):

    def setUp(self):

        self.y_true = np.array([0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0])
        self.y_pred = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0])
        self.y_pred2 =np.array([0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,0,0])
        self.y_pred3 = np.zeros(len(self.y_true))


    def test(self):
        metric = BalancedPointadjustedFScore(w=0)
        f_score = round(metric.compute(self.y_true, self.y_pred),2)
        expected_f_score = 0.9
        self.assertAlmostEqual(f_score, expected_f_score, places=4)
        metric = BalancedPointadjustedFScore(w=1)
        f_score = round(metric.compute(self.y_true, self.y_pred2),2)
        expected_f_score = 0.9
        self.assertAlmostEqual(f_score, expected_f_score, places=4)

        f_score = round(metric.compute(self.y_true, self.y_pred3),2)
        expected_f_score = 0
        self.assertAlmostEqual(f_score, expected_f_score, places=4)
        
    def test_consistency(self):
        metric = BalancedPointadjustedFScore()
        try:
            y_true = np.random.choice([0, 1], size=(100,))
            y_pred = np.zeros(100)
            metric.compute(y_true, y_pred)
            for _ in range(1000):
                y_true = np.random.choice([0, 1], size=(100,))
                y_pred = np.random.choice([0, 1], size=(100,))
                f_score = metric.compute(y_true, y_pred)
        except Exception as e:
            self.fail(f"BalancedPointadjustedFScore raised an exception {e}")


class TestCompositeFScore(unittest.TestCase):

    def setUp(self):
        """
        Configuración inicial para las pruebas.
        """
        self.y_true  = np.array([0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1])
        self.y_pred1 = np.array([0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        self.y_pred2 = np.array([0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0])
        self.y_pred3 = np.array([0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1])
        self.y_pred4 = np.zeros(len(self.y_true))




    def test(self):

        metric = CompositeFScore()
        f_score = round(metric.compute(self.y_true, self.y_pred1),2)
        expected_f_score = 0.67
        self.assertAlmostEqual(f_score, expected_f_score, places=4)

        f_score = round(metric.compute(self.y_true, self.y_pred2),2)
        expected_f_score = 1
        self.assertAlmostEqual(f_score, expected_f_score, places=4)

        f_score = round(metric.compute(self.y_true, self.y_pred3),2)
        expected_f_score = 1
        self.assertAlmostEqual(f_score, expected_f_score, places=4)

        f_score = round(metric.compute(self.y_true, self.y_pred4),2)
        expected_f_score = 0
        self.assertAlmostEqual(f_score, expected_f_score, places=4)

        
    def test_consistency(self):
        metric = CompositeFScore()
        try:
            y_true = np.random.choice([0, 1], size=(100,))
            y_pred = np.zeros(100)
            metric.compute(y_true, y_pred)
            for _ in range(1000):
                y_true = np.random.choice([0, 1], size=(10,))
                y_pred = np.random.choice([0, 1], size=(10,))
                f_score = metric.compute(y_true, y_pred)
                
        except Exception as e:
            self.fail(f"CompositeFScore raised an exception {e}")

class TestPointadjustedFScore(unittest.TestCase):

    def setUp(self):

        self.y_true = np.array([0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0])
        self.y_pred = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0])
        self.y_pred2 = np.array([0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0])
        self.y_pred3 = np.zeros(len(self.y_true))


    def test(self):
        metric = PointadjustedFScore()
        f_score = round(metric.compute(self.y_true, self.y_pred),2)
        expected_f_score = 0.93
        self.assertAlmostEqual(f_score, expected_f_score, places=4)

        f_score = round(metric.compute(self.y_true, self.y_pred2),2)
        expected_f_score = 1
        self.assertAlmostEqual(f_score, expected_f_score, places=4)

        f_score = round(metric.compute(self.y_true, self.y_pred3),2)
        expected_f_score = 0
        self.assertAlmostEqual(f_score, expected_f_score, places=4)
        
    def test_consistency(self):
        metric = PointadjustedFScore()
        try:
            y_true = np.random.choice([0, 1], size=(100,))
            y_pred = np.zeros(100)
            metric.compute(y_true, y_pred)
            for _ in range(1000):
                y_true = np.random.choice([0, 1], size=(100,))
                y_pred = np.random.choice([0, 1], size=(100,))
                f_score = metric.compute(y_true, y_pred)
        except Exception as e:
            self.fail(f"PointadjustedFScore raised an exception {e}")




class TestSegmentwiseFScore(unittest.TestCase):

    def setUp(self):

        self.y_true  = np.array([0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1])
        self.y_pred1 = np.array([0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        self.y_pred2 = np.array([0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0])
        self.y_pred3 = np.array([0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1])
        self.y_pred4 = np.zeros(len(self.y_true))


    def test(self):
        metric = SegmentwiseFScore()
        f_score = round(metric.compute(self.y_true, self.y_pred1),2)
        expected_f_score = 0.67
        self.assertAlmostEqual(f_score, expected_f_score, places=4)

        f_score = round(metric.compute(self.y_true, self.y_pred2),2)
        expected_f_score = 1
        self.assertAlmostEqual(f_score, expected_f_score, places=4)

        f_score = round(metric.compute(self.y_true, self.y_pred3),2)
        expected_f_score = 1
        self.assertAlmostEqual(f_score, expected_f_score, places=4)

        f_score = round(metric.compute(self.y_true, self.y_pred4),2)
        expected_f_score = 0
        self.assertAlmostEqual(f_score, expected_f_score, places=4)

    
        
    def test_consistency(self):
        metric = SegmentwiseFScore()
        try:
            y_true = np.random.choice([0, 1], size=(100,))
            y_pred = np.zeros(100)
            metric.compute(y_true, y_pred)
            for _ in range(1000):
                y_true = np.random.choice([0, 1], size=(10,))
                y_pred = np.random.choice([0, 1], size=(10,))
                f_score = metric.compute(y_true, y_pred)
                
        except Exception as e:
            self.fail(f"SegmentwiseFScore raised an exception {e}")



class TestPointadjustedAucRoc(unittest.TestCase):

    def setUp(self):

 
        self.y_true1 =  np.array([0,0,1,1])


        self.y_pred1 = np.array([1, 3, 2, 4])

        self.y_pred2 = np.array([1, 2, 3, 4])

        self.y_pred3 = np.array([4, 4, 4, 4])

        self.y_true2 = np.array([0,1,1,0,0,0,0,0,1,1,0,0,0,0,1,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,1,0,0,1,1,0
        ,1,1,1,0,0,1,0,0,1,0,1,1,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,1,0,1,1,1,1,1,0,1,1
        ,1,1,1,1,0,0,1,1,1,1,0,1,0,0,1,1,1,0,0,1,0,0,1,0,1,1])


        self.y_pred4 = [0.1280475, 0.12059283 ,0.29936968 ,0.85866402 ,0.74071874 ,0.22310849
        ,0.11281839 ,0.26133246 ,0.33696106 ,0.01442675 ,0.51962876 ,0.07828833
        ,0.45337844 ,0.09444483 ,0.91216588 ,0.18847595 ,0.26828481 ,0.65248919
        ,0.46291981 ,0.43730757 ,0.78087553 ,0.45031043 ,0.88661033 ,0.56209352
        ,0.45029423 ,0.17638205 ,0.9261279 ,0.58830652 ,0.01602648 ,0.73903379
        ,0.61831379 ,0.74779903 ,0.42682106 ,0.82583519 ,0.19709012 ,0.44925962
        ,0.62752415 ,0.52458327 ,0.46291768 ,0.33937527 ,0.34868777 ,0.12293847
        ,0.84477504 ,0.10225254 ,0.37048167 ,0.04476031 ,0.36680499 ,0.11346155
        ,0.10583112 ,0.09493136 ,0.54878736 ,0.68514489 ,0.5940307 ,0.14526962
        ,0.69385728 ,0.38888727 ,0.61495304 ,0.06795402 ,0.02894603 ,0.08293609
        ,0.22865685 ,0.63531487 ,0.97966126 ,0.31418622 ,0.8943095 ,0.22974177
        ,0.94402929 ,0.13140625 ,0.80539267 ,0.40160344 ,0.38151339 ,0.65011626
        ,0.71657942 ,0.93297398 ,0.32043329 ,0.54667941 ,0.90645979 ,0.98730183
        ,0.82351336 ,0.10404812 ,0.6962921 ,0.72890752 ,0.49700666 ,0.47461103
        ,0.59696079 ,0.85876179 ,0.247344 ,0.38187879 ,0.23906861 ,0.5266315
        ,0.08171512 ,0.27903375 ,0.61112439 ,0.20784267 ,0.90652453 ,0.87575255
        ,0.26972245 ,0.78780138 ,0.37649185 ,0.08467683]

        self.y_pred5 = self.y_true1
        self.y_pred6 = np.zeros(len(self.y_true1))


    def test(self):
        metric = PointadjustedAucRoc()
        score = round(metric.compute(self.y_true1, self.y_pred1),2)
        expected_score = 1.0
        self.assertAlmostEqual(score, expected_score, places=4)

        score = round(metric.compute(self.y_true1, self.y_pred2),2)
        expected_score = 1.0
        self.assertAlmostEqual(score, expected_score, places=4)

        score = round(metric.compute(self.y_true1, self.y_pred3),2)
        expected_score = 0.5
        self.assertAlmostEqual(score, expected_score, places=4)

        
        score = round(metric.compute(self.y_true2, self.y_pred4),2)
        expected_score = 0.75
        self.assertAlmostEqual(score, expected_score, places=4)

        score = round(metric.compute(self.y_true1, self.y_pred5),2)
        expected_score = 1.0
        self.assertAlmostEqual(score, expected_score, places=4)

        score = round(metric.compute(self.y_true1, self.y_pred6),2)
        expected_score = 0.5
        self.assertAlmostEqual(score, expected_score, places=4)



        
    def test_consistency(self):
        y_true, y_pred = [],[]
        metric = PointadjustedAucRoc()
        try:
            for _ in range(100):
                y_true = np.random.choice([0, 1], size=(100,))
                y_pred = np.random.random( size=(100,))
                score = metric.compute(y_true, y_pred)
        except Exception as e:
            self.fail(f"PointadjustedAucRoc raised an exception {e}")



class TestPointadjustedAucPr(unittest.TestCase):

    def setUp(self):
        """
        Configuración inicial para las pruebas.
        """
 
        self.y_true1 =  np.array([0,0,1,1])


        self.y_pred1 = np.array([1, 3, 2, 4])

        self.y_pred2 = np.array([1, 2, 3, 4])

        self.y_pred3 = np.array([4, 4, 4, 4])

        self.y_true2 = np.array([0,1,1,0,0,0,0,0,1,1,0,0,0,0,1,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,1,0,0,1,1,0
        ,1,1,1,0,0,1,0,0,1,0,1,1,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,1,0,1,1,1,1,1,0,1,1
        ,1,1,1,1,0,0,1,1,1,1,0,1,0,0,1,1,1,0,0,1,0,0,1,0,1,1])


        self.y_pred4 = [0.1280475, 0.12059283 ,0.29936968 ,0.85866402 ,0.74071874 ,0.22310849
        ,0.11281839 ,0.26133246 ,0.33696106 ,0.01442675 ,0.51962876 ,0.07828833
        ,0.45337844 ,0.09444483 ,0.91216588 ,0.18847595 ,0.26828481 ,0.65248919
        ,0.46291981 ,0.43730757 ,0.78087553 ,0.45031043 ,0.88661033 ,0.56209352
        ,0.45029423 ,0.17638205 ,0.9261279 ,0.58830652 ,0.01602648 ,0.73903379
        ,0.61831379 ,0.74779903 ,0.42682106 ,0.82583519 ,0.19709012 ,0.44925962
        ,0.62752415 ,0.52458327 ,0.46291768 ,0.33937527 ,0.34868777 ,0.12293847
        ,0.84477504 ,0.10225254 ,0.37048167 ,0.04476031 ,0.36680499 ,0.11346155
        ,0.10583112 ,0.09493136 ,0.54878736 ,0.68514489 ,0.5940307 ,0.14526962
        ,0.69385728 ,0.38888727 ,0.61495304 ,0.06795402 ,0.02894603 ,0.08293609
        ,0.22865685 ,0.63531487 ,0.97966126 ,0.31418622 ,0.8943095 ,0.22974177
        ,0.94402929 ,0.13140625 ,0.80539267 ,0.40160344 ,0.38151339 ,0.65011626
        ,0.71657942 ,0.93297398 ,0.32043329 ,0.54667941 ,0.90645979 ,0.98730183
        ,0.82351336 ,0.10404812 ,0.6962921 ,0.72890752 ,0.49700666 ,0.47461103
        ,0.59696079 ,0.85876179 ,0.247344 ,0.38187879 ,0.23906861 ,0.5266315
        ,0.08171512 ,0.27903375 ,0.61112439 ,0.20784267 ,0.90652453 ,0.87575255
        ,0.26972245 ,0.78780138 ,0.37649185 ,0.08467683]

        self.y_pred5 = self.y_true1
        self.y_pred6 = np.zeros(len(self.y_true1))

    def test(self):
        metric = PointadjustedAucPr()
        score = round(metric.compute(self.y_true1, self.y_pred1),2)
        expected_score = 1.0
        self.assertAlmostEqual(score, expected_score, places=4)

        score = round(metric.compute(self.y_true1, self.y_pred2),2)
        expected_score =  1.0
        self.assertAlmostEqual(score, expected_score, places=4)

        score = round(metric.compute(self.y_true1, self.y_pred3),2)
        expected_score = 0.75
        self.assertAlmostEqual(score, expected_score, places=4)

        
        score = round(metric.compute(self.y_true2, self.y_pred4),2)
        expected_score = 0.78
        self.assertAlmostEqual(score, expected_score, places=4)

        score = round(metric.compute(self.y_true1, self.y_pred5),2)
        expected_score = 1
        self.assertAlmostEqual(score, expected_score, places=4)

        score = round(metric.compute(self.y_true1, self.y_pred6),2)
        expected_score = 0.75
        self.assertAlmostEqual(score, expected_score, places=4)

        
    def test_consistency(self):
        y_true, y_pred = [],[]
        metric = PointadjustedAucPr()
        try:
            for _ in range(100):
                y_true = np.random.choice([0, 1], size=(100,))
                y_pred = np.random.random( size=(100,))
                score = metric.compute(y_true, y_pred)
        except Exception as e:
            self.fail(f"PointadjustedAucPr raised an exception {e}")




class TestRangebasedFScore(unittest.TestCase):

    def setUp(self):

        self.y_true1 = np.array([0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1])
        self.y_pred1 = np.array([0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        self.y_pred2 = np.array([0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0])

        self.y_true2  = np.array([0,0,1,0,1,0,1,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0])
        self.y_pred21 = np.array([0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        self.y_pred22 = np.array([0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0])

        self.y_pred3 = np.array([0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1])
        self.y_pred4 = np.zeros(len(self.y_true1))
    

    def test(self):
        metric = RangebasedFScore(beta=1,p_alpha=0.2,r_alpha=0.2,cardinality_mode='one',p_bias='flat',r_bias='flat')
        f_score = round(metric.compute(self.y_true1, self.y_pred1),2)
        expected_f_score = 0.67
        self.assertAlmostEqual(f_score, expected_f_score, places=4)

        f_score = round(metric.compute(self.y_true1, self.y_pred2),2)
        expected_f_score = 0.46
        self.assertAlmostEqual(f_score, expected_f_score, places=4)

        f_score = round(metric.compute(self.y_true2, self.y_pred21),2)
        expected_f_score = 0.71
        self.assertAlmostEqual(f_score, expected_f_score, places=4)

        f_score = round(metric.compute(self.y_true2, self.y_pred22),2)
        expected_f_score = 0.67
        self.assertAlmostEqual(f_score, expected_f_score, places=4)

        f_score = round(metric.compute(self.y_true1, self.y_pred3),2)
        expected_f_score = 1
        self.assertAlmostEqual(f_score, expected_f_score, places=4)

        f_score = round(metric.compute(self.y_true1, self.y_pred4),2)
        expected_f_score = 0
        self.assertAlmostEqual(f_score, expected_f_score, places=4)
        
    def test_range_based_consistency(self):
        
        try:
            modes = ['flat','front','back'
                     ,'middle']
            modes_c = ['one','reciprocal']
            metric = RangebasedFScore(beta=2,p_alpha=random.random(),r_alpha=random.random(),cardinality_mode=random.choice(modes_c),p_bias=random.choice(modes),r_bias=random.choice(modes))
            y_true = np.random.choice([0, 1], size=(100,))
            y_pred = np.zeros(100)
            metric.compute(y_true, y_pred)
            for _ in range(100):
                y_true = np.random.choice([0, 1], size=(100,))
                y_pred = np.random.choice([0, 1], size=(100,))
                metric = RangebasedFScore(beta=2,p_alpha=random.random(),r_alpha=random.random(),cardinality_mode=random.choice(modes_c),p_bias=random.choice(modes),r_bias=random.choice(modes))
                f_score = metric.compute(y_true, y_pred)
        except Exception as e:
            self.fail(f"RangeBasedFScore raised an exception {e}")
