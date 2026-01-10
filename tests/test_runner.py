import unittest

from tsadmetrics.evaluation.Runner import Runner
import numpy as np
import os

class TestRunner(unittest.TestCase):


    def setUp(self):
        """
        Configuración inicial para las pruebas.
        """
        self.y_true1 = np.array([0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1])
        self.y_pred1 = np.array([0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        self.y_pred2 = np.array([0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0])

        

    

    def test_direct_data(self):
        dataset_evaluations = [
            ("dataset1", self.y_true1, (self.y_pred1, self.y_pred1)),
            ("dataset2", self.y_true1, (self.y_pred2, self.y_pred2))

        ]
        metrics = [
            ("adc",{}),
            ("dair",{}),
            ("pakf",{"k":0.2}),]
        expected_score_d1 = {
            "adc":0.5,
            "dair":1.0,
            "pakf":0.67
        }
        expected_score_d2 = {
            "adc":0.12,
            "dair":1.0,
            "pakf":0.22
        }
        runnner = Runner(dataset_evaluations, metrics)
        results = runnner.run()
        for metric, expected in expected_score_d1.items():
            params = {}
            for p in metrics:
                if p[0] == metric:
                    params = p[1]
            print(metric)
            self.assertAlmostEqual(round(results.loc["dataset1", f"{metric}"], 2), expected, places=4)
        for metric, expected in expected_score_d2.items():
            params = {}
            for p in metrics:
                if p[0] == metric:
                    params = p[1]
            self.assertAlmostEqual(round(results.loc["dataset2", f"{metric}"],2), expected, places=4)

    def test_file_reference(self):
        dataset_evaluations = [
            ("dataset1", "tests/test_data/results1.csv"),
            ("dataset2", "tests/test_data/results2.csv")

        ]
        metrics = [
            ("adc",{}),
            ("dair",{}),
            ("pakf",{"k":0.2})]
        expected_score_d1 = {
            "adc":0.5,
            "dair":1.0,
            "pakf":0.67
        }
        expected_score_d2 = {
            "adc":0.12,
            "dair":1.0,
            "pakf":0.22
        }
        runnner = Runner(dataset_evaluations, metrics)
        results = runnner.run()
        for metric, expected in expected_score_d1.items():
            params = {}
            for p in metrics:
                if p[0] == metric:
                    params = p[1]
            self.assertAlmostEqual(round(results.loc["dataset1",f"{metric}"],2), expected, places=4)
        for metric, expected in expected_score_d2.items():
            params = {}
            for p in metrics:
                if p[0] == metric:
                    params = p[1]
            self.assertAlmostEqual(round(results.loc["dataset2", f"{metric}"],2), expected, places=4)

    def test_metrics_from_file(self):
        dataset_evaluations = [
            ("dataset1", "tests/test_data/results1.csv"),
            ("dataset2", "tests/test_data/results2.csv")

        ]
        metrics = [
            ("adc",{}),
            ("dair",{}),
            ("pakf",{"k":0.2})]
        expected_score_d1 = {
            "adc":0.5,
            "dair":1.0,
            "pakf":0.67
        }
        expected_score_d2 = {
            "adc":0.12,
            "dair":1.0,
            "pakf":0.22
        }
        runnner = Runner(dataset_evaluations,"tests/test_data/example_metrics_config.yaml")
        results = runnner.run()
        for metric, expected in expected_score_d1.items():
            params = {}
            for p in metrics:
                if p[0] == metric:
                    params = p[1]
            self.assertAlmostEqual(round(results.loc["dataset1",f"{metric}"],2), expected, places=4)
        for metric, expected in expected_score_d2.items():
            params = {}
            for p in metrics:
                if p[0] == metric:
                    params = p[1]
            self.assertAlmostEqual(round(results.loc["dataset2", f"{metric}"],2), expected, places=4)

    def test_evaluation_from_file(self):
        metrics = [
            ("adc",{}),
            ("dair",{}),
            ("pakf",{"k":0.2})]
        expected_score_d1 = {
            "adc":0.5,
            "dair":1.0,
            "pakf":0.67
        }
        expected_score_d2 = {
            "adc":0.12,
            "dair":1.0,
            "pakf":0.22
        }
        runnner = Runner("tests/test_data/example_evaluation_config.yaml")
        results = runnner.run()
        print(results)
        for metric, expected in expected_score_d1.items():
            params = {}
            for p in metrics:
                if p[0] == metric:
                    params = p[1]
            self.assertAlmostEqual(round(results.loc["dataset1", f"{metric}"],2), expected, places=4)
        for metric, expected in expected_score_d2.items():
            params = {}
            for p in metrics:
                if p[0] == metric:
                    params = p[1]
            self.assertAlmostEqual(round(results.loc["dataset2",f"{metric}"],2), expected, places=4)
    def test_report(self):
        dataset_evaluations = [
            ("dataset1", "tests/test_data/results1.csv"),
            ("dataset2", "tests/test_data/results2.csv")
        ]

        metrics = [
            ("adc",{}),
            ("dair",{}),
            ("pakf",{"k":0.2})]
        expected_score_d1 = {
            "adc":0.5,
            "dair":1.0,
            "pakf":0.67
        }
        expected_score_d2 = {
            "adc":0.12,
            "dair":1.0,
            "pakf":0.22
        }
        runnner = Runner(dataset_evaluations, metrics)
        results = runnner.run(generate_report=True, report_file="tests/evaluation_report.csv")

        with open("tests/evaluation_report.csv", "r") as generated_file, open("tests/test_data/evaluation_report.csv", "r") as expected_file:
            self.assertEqual(generated_file.read(), expected_file.read())

        os.remove("tests/evaluation_report.csv")
        