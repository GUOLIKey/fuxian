import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LIVE = Path('/Users/cheeseowl/Desktop/科研/fuxian/FedVaDE_source')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class SourceSnapshotTests(unittest.TestCase):
    def test_declared_snapshot_matches_hashes_and_live_author_source(self):
        manifest_path = ROOT / 'SOURCE_MANIFEST.json'
        self.assertTrue(manifest_path.exists(), 'SOURCE_MANIFEST.json missing')
        manifest = json.loads(manifest_path.read_text())
        self.assertEqual(
            set(manifest),
            {f'vendor/author/{name}' for name in
             ('main.py', 'model.py', 'dataloader.py', 'experiment.py')},
        )
        for relative, expected in manifest.items():
            frozen = ROOT / relative
            self.assertTrue(frozen.exists(), relative)
            self.assertEqual(sha256(frozen), expected)
            if LIVE.exists():
                self.assertEqual(frozen.read_bytes(), (LIVE / frozen.name).read_bytes())

    def test_audited_common_runner_is_present(self):
        self.assertTrue((ROOT / 'vendor/audited_scripts/common/train.py').exists())
        self.assertTrue((ROOT / 'vendor/audited_scripts/common/evaluate.py').exists())


if __name__ == '__main__':
    unittest.main()
