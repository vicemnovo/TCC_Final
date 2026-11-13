"""Seleção do piloto 3.7 sem rede, Drive ou inferência."""

import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    'fase3_7_openimages_pilot', ROOT / 'Notebooks/fase_3/fase3_7/fase3_7_openimages_pilot.py')
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


def row(image_id, author, *, license_url='https://creativecommons.org/licenses/by/2.0/'):
    return {'ImageID': image_id, 'Subset': 'train',
            'OriginalURL': f'https://staticflickr.com/{image_id}.jpg',
            'OriginalLandingURL': f'https://www.flickr.com/photos/{author}/{image_id}',
            'License': license_url,
            'AuthorProfileURL': f'https://www.flickr.com/people/{author}/',
            'Author': author, 'OriginalSize': '500000', 'OriginalMD5': ''}


class OpenImagesPilotTests(unittest.TestCase):
    def test_aceita_foto_com_autor_e_licenca_by(self):
        result = pilot.eligible(row('0123456789abcdef', 'fotografo'))
        self.assertEqual(result['author_group'], 'flickr:fotografo')
        self.assertEqual(result['image_id'], '0123456789abcdef')

    def test_rejeita_origem_ou_licenca_sem_rastreio(self):
        original = row('0123456789abcdef', 'autor')
        for changed in ({'AuthorProfileURL': ''}, {'Subset': 'test'},
                        {'License': 'https://creativecommons.org/licenses/by-nc/2.0/'},
                        {'OriginalLandingURL': 'https://example.org/photo'},
                        {'ImageID': '../foto'}):
            with self.subTest(changed=changed):
                self.assertIsNone(pilot.eligible({**original, **changed}))

    def test_piloto_deterministico_e_autores_distintos(self):
        with tempfile.TemporaryDirectory() as temporary:
            metadata = Path(temporary) / 'metadata.csv'
            with metadata.open('w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=sorted(pilot.REQUIRED_COLUMNS))
                writer.writeheader()
                for index in range(30):
                    writer.writerow(row(f'{index:016x}', f'autor{index % 6}'))
            first = pilot.scan_metadata(metadata, 3)
            second = pilot.scan_metadata(metadata, 3)
            self.assertEqual(first, second)
            self.assertEqual(first['eligible_images'], 30)
            self.assertEqual(first['distinct_authors'], 6)
            self.assertEqual(len({item['author_group'] for item in first['pilot_rows']}), 3)
            self.assertEqual(len({item['image_id'] for item in first['pilot_rows']}), 3)

    def test_csv_incompleto_falha_antes_de_selecionar(self):
        with tempfile.TemporaryDirectory() as temporary:
            metadata = Path(temporary) / 'metadata.csv'
            metadata.write_text('ImageID,Subset\n0123456789abcdef,train\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Colunas oficiais'):
                pilot.scan_metadata(metadata)

    def test_notebook_exige_piloto_manual_sem_treino(self):
        path = ROOT / 'Notebooks/fase_3/fase3_7/piloto_fase3_7_openimages_colab.ipynb'
        notebook = json.loads(path.read_text(encoding='utf-8'))
        sources = [''.join(cell['source']) for cell in notebook['cells']]
        for index, cell in enumerate(notebook['cells']):
            if cell['cell_type'] == 'code':
                compile(sources[index], f'cell{index + 1}', 'exec')
                self.assertIsNone(cell['execution_count'])
                self.assertFalse(cell['outputs'])
        self.assertEqual(sum("run_live('piloto Open Images 3.7'" in source
                             for source in sources), 1)
        self.assertIn('Notebooks/fase_3/fase3_7/fase3_7_openimages_pilot.py', ''.join(sources))
        self.assertNotIn('fase3_5_finetune.py', ''.join(sources))


if __name__ == '__main__':
    unittest.main()
