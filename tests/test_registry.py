import unittest
from tsadmetrics.metrics.Registry import Registry
from sklearn.metrics import fbeta_score
import numpy as np
import random
class TestRegistry(unittest.TestCase):
    def setUp(self):
        """
        Configuración inicial para las pruebas.
        """
        self.registry = Registry()
        self.sample_metric = "pwf"
        
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
                metric = self.registry.get_metric(self.sample_metric,beta=beta)
                f_score = metric.compute(y_true, y_pred)
                expected_f_score = fbeta_score(y_true, y_pred, beta=beta)
                self.assertAlmostEqual(f_score, expected_f_score, places=4)

    def test_load_metrics_from_file(self):
        """
        Prueba que las métricas se pueden cargar desde un fichero.
        """
        metrics_file = "tests/test_data/example_metrics_config.yaml"  


        loaded_metrics = self.registry.load_metrics_info_from_file(metrics_file)

        expected_metrics = [
            ("adc", {}),
            ("dair", {}),
            ("pakf", {"k":0.2})
        ]
        self.assertEqual(len(loaded_metrics), len(expected_metrics))
        for (m1, p1), (m2, p2) in zip(loaded_metrics, expected_metrics):
            self.assertEqual(m1, m2)
            self.assertEqual(p1, p2)
    
