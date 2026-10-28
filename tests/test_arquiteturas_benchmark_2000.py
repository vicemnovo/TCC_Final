"""Regressão estática para a extração das arquiteturas históricas."""

import ast
import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "avaliar_benchmark_2000", ROOT / "experimentos/avaliar_benchmark_2000.py"
)
# O import do avaliador requer o módulo irmão, mas não PyTorch no teste estático.
import sys
sys.path.insert(0, str(ROOT / "experimentos"))
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class ArchitectureSourceTests(unittest.TestCase):
    def test_ambas_as_classes_originais_sao_extraidas_sem_treinar(self):
        phase1, phase2, digest = module.architecture_sources(ROOT)
        phase1_names = {node.name for node in phase1.body if isinstance(node, ast.ClassDef)}
        phase2_names = {
            node.name for node in ast.parse(phase2).body if isinstance(node, ast.ClassDef)
        }
        self.assertIn("Phase1Detector", phase1_names)
        self.assertIn("Phase2Detector", phase2_names)
        self.assertNotIn("model = Phase1Detector()", ast.unparse(phase1))
        self.assertEqual(len(digest), 64)
        compile(phase1, "phase1", "exec")
        compile(phase2, "phase2", "exec")


if __name__ == "__main__":
    unittest.main()
