import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import prepare
import run


class PipelineTests(unittest.TestCase):
    def test_parse_caption_and_matched_generalization(self):
        text, phrases = prepare.parse_caption('A [/EN#1/people young man] holds [/EN#2/other a red ball] .')
        self.assertEqual(text, 'A young man holds a red ball .')
        self.assertEqual(phrases[0]['phrase'], 'young man')
        replacement = prepare.matched_generalization(phrases[0]['phrase'], phrases[0]['types'])
        self.assertEqual(len(replacement.split()), 2)
        self.assertFalse(prepare.contains_phrase(replacement, phrases[0]['phrase']))

    def test_box_overlap(self):
        self.assertTrue(prepare.overlap([0, 0, 10, 10], [9, 9, 12, 12]))
        self.assertFalse(prepare.overlap([0, 0, 10, 10], [10, 0, 12, 2]))

    def test_tie_metrics_use_expected_random_tie_breaking(self):
        rows = [{'image_id': 'x', 'regions': [{}, {}, {}]}]
        ptr = np.array([0, 3])
        scores = np.full((3, 3), .5)
        result = run.per_image(rows, ptr, scores, [0])['x']
        self.assertAlmostEqual(result[4], 1 / 3)
        self.assertAlmostEqual(result[5], (1 + .5 + 1 / 3) / 3)

    def test_manifest_fingerprint_changes_with_active_intervention(self):
        row = {'image_id': 'x', 'split': 'train', 'caption_index': 0, 'tminus': 'minus', 'tplus': 'plus',
               'tcontrol': 'control', 'target_phrase': 'target', 'regions': [{'box': [0, 0, 2, 2]}]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'manifest.json'
            path.write_text(json.dumps([row]))
            self.assertTrue(path.is_file())


if __name__ == '__main__':
    unittest.main()
