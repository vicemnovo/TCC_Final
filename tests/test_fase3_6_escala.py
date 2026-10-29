"""Guarda o isolamento experimental e a interface manual da Fase 3.6."""

import ast
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'experimentos/fase3_5_finetune.py'
VARIANT = ROOT / 'experimentos/fase3_6_escala.py'
NOTEBOOK = ROOT / 'Notebooks/fase_3/rodar_fase3_6_colab.ipynb'


def assignment(tree, name):
    return next(node.value for node in tree.body if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == name
                        for target in node.targets))


class Fase36IsolationTests(unittest.TestCase):
    def test_mesmos_dados_inicio_e_apenas_um_aumento_alterado(self):
        base = ast.parse(BASE.read_text(encoding='utf-8'))
        variant = ast.parse(VARIANT.read_text(encoding='utf-8'))
        for name in ('START', 'OLD_MANIFEST', 'NEW_LOCAL', 'NEW_MANIFEST'):
            self.assertEqual(ast.dump(assignment(base, name)),
                             ast.dump(assignment(variant, name)))
        self.assertEqual(ast.literal_eval(assignment(variant, 'RUN_NAME')),
                         'fase3_6_escala_p05_v1')
        source = VARIANT.read_text(encoding='utf-8')
        self.assertIn("CFG['SCALE_AUGMENT_P'] = 0.50", source)
        self.assertIn("RandomDownUp(CFG['SCALE_AUGMENT_P'])", source)
        self.assertIn("Partições da 3.6 diferem das da 3.5", source)
        self.assertIn("START = ROOT / 'modelos/fase3_3_compressao_v1/phase3_final.pt'", source)

    def test_notebook_nao_inicia_treino_ao_abrir(self):
        notebook = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
        cells = [''.join(cell['source']) for cell in notebook['cells']]
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                compile(''.join(cell['source']), '<cell>', 'exec')
        # O usuário já executou o notebook: contadores e saídas podem existir.
        # O isolamento depende das chamadas nas células, não desses metadados.
        self.assertIn('experimentos/fase3_6_escala.py', cells[5])
        self.assertEqual(sum("'auto'" in cell for cell in cells), 1)
        self.assertIn("'smoke'", cells[8])
        self.assertIn("'auto'", cells[9])
        self.assertFalse(any("'auto'" in cell for cell in cells[:9]))


if __name__ == '__main__':
    unittest.main()
