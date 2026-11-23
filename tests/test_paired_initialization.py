"""Checks for the deterministic warm-start snapshots. No training."""
from __future__ import annotations

import hashlib
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

EXP = Path('/root/autodl-tmp/fedvade_project/experiments/multica_pretrained_2x2_v1')


def tensor_bytes(x):
    if hasattr(x, 'detach'):
        x = x.detach().cpu().numpy()
    return np.asarray(x).tobytes()


def state_hash(state, exclude_gmm=False):
    h = hashlib.sha256()
    for k in sorted(state):
        if exclude_gmm and k in ('rho_', 'mu_c', 'log_sigma2_c'):
            continue
        h.update(k.encode()); h.update(tensor_bytes(state[k]))
    return h.hexdigest()


class TestPairedInitialization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads((EXP / 'init_manifest.json').read_text())

    def test_all_seeds_present(self):
        for model in ('AUTHOR', 'F01'):
            self.assertEqual(set(self.manifest['models'][model]['seeds']), {'0', '1', '2'})

    def test_paired_client_and_encoder_hashes(self):
        for model in ('AUTHOR', 'F01'):
            for seed in ('0', '1', '2'):
                s = self.manifest['models'][model]['seeds'][seed]
                self.assertEqual(s['server_encoder_hash_off'], s['server_encoder_hash_on'])
                base = torch.load(s['base'], map_location='cpu')
                on = torch.load(s['on'], map_location='cpu')
                self.assertEqual(base['client_hash'], on['client_hash'])
                self.assertEqual(state_hash(base['clients'][0]), state_hash(on['clients'][0]))
                self.assertEqual(base['data_seed'], on['data_seed'])
                self.assertEqual(base['local_K'], on['local_K'])

    def test_true_micro_capacity(self):
        for model in ('AUTHOR', 'F01'):
            m = self.manifest['models'][model]
            self.assertGreater(m['K_micro'], 2)
            for seed in ('0', '1', '2'):
                on = torch.load(m['seeds'][seed]['on'], map_location='cpu')
                self.assertEqual(int(on['server_state']['rho_'].numel()), m['K_micro'])
                self.assertEqual(sum(on['macro_allocation']), m['K_micro'])
                self.assertGreaterEqual(min(on['macro_allocation']), 2)
                self.assertEqual(len(on['macro_of_micro']), m['K_micro'])

    def test_off_server_stays_k2(self):
        for model in ('AUTHOR', 'F01'):
            for seed in ('0', '1', '2'):
                base = torch.load(self.manifest['models'][model]['seeds'][seed]['base'],
                                  map_location='cpu')
                self.assertEqual(int(base['server_state']['rho_'].numel()), 2)

    def test_initial_metrics_present(self):
        for model in ('AUTHOR', 'F01'):
            m = self.manifest['models'][model]
            self.assertGreater(m['source_initial_metrics']['ari']['mean'], 0.4)
            for seed in ('0', '1', '2'):
                self.assertGreater(m['seeds'][seed]['initial_metrics_off']['ari']['mean'], 0.4)


if __name__ == '__main__':
    unittest.main()
