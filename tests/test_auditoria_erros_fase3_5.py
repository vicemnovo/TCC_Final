"""Auditoria dos erros internos com arquivos de teste pequenos, sem GPU."""

import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

from PIL import Image


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experimentos'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'Notebooks/fase_3/fase3_5'))
from auditar_erros_internos_fase3_5 import audit, sha256, validate_predictions


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding='utf-8')


class AuditoriaErrosTest(unittest.TestCase):
    def test_rejeita_predicao_que_nao_aplica_limiar_congelado(self):
        selection = [{'path': 'a.jpg', 'label': 1, 'group': 'gerador'}]
        prediction = [{'subset': 'validation_selection', 'path': 'a.jpg',
                       'label': '1', 'group': 'gerador', 'probability_fake': '0.1',
                       'prediction': '1', 'correct': '1'}]
        with self.assertRaisesRegex(ValueError, 'limiar'):
            validate_predictions(prediction, selection, 0.445)

    def test_audita_somente_a_selecao_e_preserva_resultados(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'drive'
            local = Path(temporary) / 'content'
            old_img = root / 'dados_fase3_4/real.jpg'
            new_img = local / 'images/1/fake.jpg'
            for image_path in (old_img, new_img):
                image_path.parent.mkdir(parents=True, exist_ok=True)
                Image.new('RGB', (640, 480), 'white').save(image_path, format='JPEG')
            old_row = {'path': str(old_img), 'label': 0, 'group': 'pexels',
                       'orig_format': 'PNG', 'orig_w': 640, 'orig_h': 480}
            new_row = {'path': str(new_img), 'label': 1, 'group': 'cfsmall:generator'}
            write_json(root / 'dados_fase3_4/manifest.json',
                       {'train': [old_row], 'test': []})
            write_json(local / 'manifest.json',
                       {'rows': [{'image': 'images/1/fake.jpg', 'label': 1,
                                  'group': 'generator'}]})
            run = root / 'modelos/fase3_5_cfsmall_v1'
            run.mkdir(parents=True)
            (run / 'phase3_final.pt').write_bytes(b'checkpoint')
            split_path = run / 'split_manifest.json'
            write_json(split_path, {'validation_selection': [old_row, new_row],
                                    'validation_calibration': [], 'test_reserved': []})
            diagnostic = root / 'resultados/diagnostico_interno_fase3_5_v1'
            diagnostic.mkdir(parents=True)
            write_json(diagnostic / 'summary.json',
                       {'external_used': False, 'calibration_inferred': False,
                        'openfake_test_inferred': False,
                        'checkpoint_sha256': sha256(run / 'phase3_final.pt'),
                        'split_manifest_sha256': sha256(split_path),
                        'threshold_frozen': 0.445,
                        'partitions': {'validation_selection': {'n': 2, 'fp': 1, 'fn': 1}}})
            with (diagnostic / 'predictions.csv').open('w', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, fieldnames=(
                    'subset', 'path', 'label', 'group', 'probability_fake', 'prediction', 'correct'))
                writer.writeheader()
                writer.writerows([
                    {'subset': 'validation_selection', 'path': str(old_img),
                     'label': 0, 'group': 'pexels', 'probability_fake': 0.9,
                     'prediction': 1, 'correct': 0},
                    {'subset': 'validation_selection', 'path': str(new_img),
                     'label': 1, 'group': 'cfsmall:generator', 'probability_fake': 0.1,
                     'prediction': 0, 'correct': 0}])
            audit(root, local)
            output = root / 'resultados/auditoria_erros_internos_fase3_5_v1'
            summary = json.loads((output / 'summary.json').read_text(encoding='utf-8'))
            self.assertEqual((summary['false_positives'], summary['false_negatives']), (1, 1))
            with (output / 'errors.csv').open(newline='', encoding='utf-8') as handle:
                errors = list(csv.DictReader(handle))
            self.assertEqual({row['dataset'] for row in errors},
                             {'OpenFake', 'CommunityForensics-Small'})
            self.assertEqual(len(errors), 2)


if __name__ == '__main__':
    unittest.main()
