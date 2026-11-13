"""Evita regressão nas mensagens de proveniência dos pesos da v2."""

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class CheckpointLogTest(unittest.TestCase):
    def test_log_distingue_arquitetura_de_pesos_ativos(self):
        script = (ROOT / 'Notebooks/fase_3/fase3_4/fase3_4_finetune_v2.py').read_text(encoding='utf-8')
        notebook = json.loads((ROOT / 'Notebooks/fase_3/fase3_3/fase3.3_compressao.ipynb').read_text(encoding='utf-8'))
        source = ''.join(notebook['cells'][49]['source'])
        self.assertIn('Fase 2 carregada |', source)
        self.assertIn("src.replace(old_message, 'Arquitetura da Fase 2 montada |', 1)", script)
        load = script.index("model.load_state_dict(start['model_state_dict'], strict=True)")
        confirmation = script.index("print(f'Pesos da Fase 3.3 carregados")
        self.assertLess(load, confirmation)
        self.assertLess(confirmation, script.index("if args.mode == 'smoke':"))


if __name__ == '__main__':
    unittest.main()
