import unittest
import pandas as pd
from failure_model import generate_synthetic_data, build_pipeline, train_and_evaluate

class TestFailureModel(unittest.TestCase):
    def test_generate_synthetic_data(self):
        n_rows = 100
        df = generate_synthetic_data(n_rows=n_rows)
        self.assertEqual(len(df), n_rows)
        self.assertIn("failure", df.columns)
        self.assertIn("material_type", df.columns)

    def test_build_pipeline(self):
        pipeline = build_pipeline()
        self.assertIsNotNone(pipeline)
        self.assertEqual(len(pipeline.steps), 2)

    def test_train_and_evaluate(self):
        model, metrics, scored_test = train_and_evaluate(n_rows=200)
        self.assertIsNotNone(model)
        self.assertGreaterEqual(metrics.accuracy, 0.0)
        self.assertLessEqual(metrics.accuracy, 1.0)
        self.assertEqual(len(scored_test), metrics.test_rows)

if __name__ == "__main__":
    unittest.main()
