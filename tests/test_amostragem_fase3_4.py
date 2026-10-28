"""Testes locais da política de amostragem, sem PyTorch ou dados do Drive."""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experimentos"))
from amostragem_fase3_4 import balanced_sqrt_group_weights
from particoes_fase3_4 import persist_or_verify, split_payload


class SamplerWeightsTest(unittest.TestCase):
    def test_classes_equilibradas_e_grupo_raro_menos_repetido(self):
        rows = ([{"label": 0, "group": "real-a"}] * 100
                + [{"label": 1, "group": "fake-raro"}]
                + [{"label": 1, "group": "fake-comum"}] * 100)
        weights = balanced_sqrt_group_weights(rows)
        self.assertAlmostEqual(sum(weights), 1.0)
        self.assertAlmostEqual(sum(w for r, w in zip(rows, weights) if r["label"] == 0), 0.5)
        self.assertAlmostEqual(sum(w for r, w in zip(rows, weights) if r["label"] == 1), 0.5)
        rare_mass = weights[100]
        common_mass = sum(weights[101:])
        self.assertAlmostEqual(common_mass / rare_mass, 10.0)
        self.assertLess(rare_mass, 0.25)  # a v1 daria 0,25 à única imagem

    def test_rejeita_classe_ausente(self):
        with self.assertRaisesRegex(ValueError, "duas classes|classes real"):
            balanced_sqrt_group_weights([{"label": 1, "group": "unico"}])

    def test_particoes_nao_sao_sobrescritas(self):
        row = {"path": "imagem.jpg", "label": 1, "group": "gerador"}
        payload = split_payload([row], [], [], [], manifest_sha256="sha",
                                sampler_policy="sqrt", replay_count=0)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "split_manifest.json"
            with self.assertRaises(FileNotFoundError):
                persist_or_verify(path, payload, resume=True)
            self.assertTrue(persist_or_verify(path, payload))
            original = path.read_bytes()
            self.assertFalse(persist_or_verify(path, payload, resume=True))
            changed = dict(payload, replay_count=1)
            with self.assertRaises(ValueError):
                persist_or_verify(path, changed, resume=True)
            self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
