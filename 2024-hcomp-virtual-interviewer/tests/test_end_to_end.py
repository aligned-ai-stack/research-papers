"""Public workflow and privacy-boundary tests; no participant data required."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import analysis
import release

class Workflows(unittest.TestCase):
    def test_public_plots_and_release_exclusion(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            subprocess.run([sys.executable, '-B', str(ROOT / 'analysis.py'), 'plots',
                            '--tables', str(ROOT / 'data'), '--output', str(tmp / 'plots')], check=True)
            self.assertEqual(len(list((tmp / 'plots').glob('*'))), 8)
            archive = tmp / 'release.zip'
            manifest = release.package(archive)
            with zipfile.ZipFile(archive) as packed:
                self.assertEqual(set(packed.namelist()), set(release.FILES) | {'SHA256SUMS.json'})
                self.assertFalse(any('review/' in name or 'local/' in name or 'PRIVATE' in name for name in packed.namelist()))
                self.assertEqual(json.loads(packed.read('SHA256SUMS.json')), manifest)
                for name, digest in manifest.items():
                    self.assertEqual(hashlib.sha256(packed.read(name)).hexdigest(), digest)

    def test_refit_workflow_and_invalid_input(self):
        rng = np.random.default_rng(15)
        n = 80
        participants = pd.DataFrame({'study_id': [f'synthetic-{i}' for i in range(n)],
                                     'age': rng.integers(20, 60, n),
                                     'user_demographic': np.tile(analysis.GROUPS, 20),
                                     'interviewer_demographic': np.tile(['A', 'B'], 40)})
        items = pd.DataFrame({'study_id': participants.study_id})
        for scale, (count, maximum) in analysis.SCALES.items():
            for i in range(1, count + 1):
                items[f'{scale}_item_{i:02d}'] = rng.integers(1, maximum + 1, n)
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            participants.to_csv(tmp / 'participants.csv', index=False)
            items.sample(frac=1, random_state=2).to_csv(tmp / 'items.csv', index=False)
            base = [sys.executable, '-B', str(ROOT / 'analysis.py'), 'run', '--participants',
                    str(tmp / 'participants.csv'), '--items', str(tmp / 'items.csv'), '--n-boot', '20']
            for profile in ['historical', 'corrected']:
                subprocess.run(base + ['--profile', profile, '--output', str(tmp / profile)], check=True,
                               capture_output=True, text=True)
            historical = pd.read_csv(tmp / 'historical/scores_PRIVATE.csv')
            corrected = pd.read_csv(tmp / 'corrected/scores_PRIVATE.csv')
            np.testing.assert_allclose(corrected.ati, historical.ati * 9)
            np.testing.assert_allclose(corrected.spp, historical.spp)
            np.testing.assert_allclose(corrected.privacy_concerns - historical.privacy_concerns,
                                       (8 - 2 * items.privacy_concerns_item_10) / 10, atol=1e-14)
            self.assertEqual(len(pd.read_csv(tmp / 'corrected/mediation.csv')), 60)
            items.loc[0, 'fairness_item_01'] = 99
            items.to_csv(tmp / 'items.csv', index=False)
            failed = subprocess.run(base + ['--profile', 'corrected', '--output', str(tmp / 'invalid')],
                                    capture_output=True, text=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn('Invalid or missing Likert response', failed.stderr)
            self.assertFalse((tmp / 'invalid').exists())

if __name__ == '__main__':
    unittest.main()
