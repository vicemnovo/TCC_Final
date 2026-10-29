"""Regressão da conferência de hashes das partições internas da Fase 3.5."""

import ast
from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / 'experimentos/fase3_5_finetune.py'


class ParticoesFase35Test(unittest.TestCase):
    def test_ordem_dos_argumentos_na_conferencia_de_hashes(self):
        source = ast.parse(SCRIPT.read_text(encoding='utf-8'))
        assignment = next(
            node for node in source.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == 'hash_sets'
                    for target in node.targets)
        )
        calls = []

        def digests(rows, title):
            self.assertIsInstance(rows, list)
            self.assertIsInstance(title, str)
            self.assertEqual(rows[0]['path'], title + '.jpg')
            calls.append(title)
            return {title}

        namespace = {
            'digests': digests,
            'tr_rows': [{'path': 'treino.jpg'}],
            'ves_rows': [{'path': 'seleção.jpg'}],
            'vcal_rows': [{'path': 'calibração.jpg'}],
        }
        expression = ast.Expression(body=assignment.value)
        result = eval(compile(expression, str(SCRIPT), 'eval'), namespace)
        self.assertEqual(calls, ['treino', 'seleção', 'calibração'])
        self.assertEqual(result, [{'treino'}, {'seleção'}, {'calibração'}])


if __name__ == '__main__':
    unittest.main()
