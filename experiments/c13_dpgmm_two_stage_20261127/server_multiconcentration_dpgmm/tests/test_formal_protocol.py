import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class FormalProtocolTests(unittest.TestCase):
    def test_one_explicit_training_seed_and_ten_inference_draws(self):
        script = (ROOT / 'scripts/run_formal.sh').read_text()
        self.assertIn('TRAINING_SEED', script)
        self.assertNotIn('for seed in 0 1 2 3 4', script)
        self.assertIn('--inference-seeds 10', script)
        self.assertIn('for seed in "$TRAINING_SEED"', script)


if __name__ == '__main__':
    unittest.main()
