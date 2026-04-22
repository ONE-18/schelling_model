import unittest
import sys
import os
# ensure `training` is on sys.path so `src` package can be imported during tests
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.schelling import SchellingModel
import random
# make tests deterministic
random.seed(0)

class TestSchellingModel(unittest.TestCase):

    def test_convergence(self):
        model = SchellingModel(num_groups=3, num_neighbors=4, board_size=10, empty_percentage=0.2, tolerance_threshold=0.3)
        result = model.init(max_generations=100)
        self.assertIn(result, [0, 1], "The result should be either 0 or 1.")

    def test_no_empty_cells(self):
        model = SchellingModel(num_groups=2, num_neighbors=4, board_size=10, empty_percentage=0.0, tolerance_threshold=0.5)
        result = model.init(max_generations=50)
        self.assertEqual(result, 1, "The model should converge with no empty cells.")

    def test_high_tolerance(self):
        model = SchellingModel(num_groups=3, num_neighbors=4, board_size=10, empty_percentage=0.1, tolerance_threshold=0.9)
        result = model.init(max_generations=100)
        self.assertEqual(result, 1, "The model should converge with high tolerance.")

    def test_low_tolerance(self):
        model = SchellingModel(num_groups=2, num_neighbors=4, board_size=10, empty_percentage=0.3, tolerance_threshold=0.1)
        result = model.init(max_generations=100)
        self.assertEqual(result, 0, "The model should not converge with low tolerance.")

if __name__ == '__main__':
    unittest.main()