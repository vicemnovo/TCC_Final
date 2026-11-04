"""Verifica o isolamento das partições da auditoria sem GPU ou datasets."""

import csv
from pathlib import Path
import sys
import tempfile
import unittest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experimentos'))
import auditoria_interna_fase3_8 as audit


class AuditoriaInternaFase38Test(unittest.TestCase):
    def test_novos_exemplos_devem_ficar_na_divisao_congelada(self):
        old = {'train': [{'path': 'old.jpg', 'label': 0, 'group': 'pexels'}],
               'test': [{'path': 'reserved.jpg', 'label': 1, 'group': 'holdout'}]}
        with tempfile.TemporaryDirectory() as temp:
            local = Path(temp)
            manifest = {'rows': [
                {'image': 'real.jpg', 'label': 0, 'group': 'camera',
                 'subset': 'validation_selection'},
                {'image': 'fake.jpg', 'label': 1, 'group': 'generator',
                 'subset': 'train'}]}
            real = {'path': str(local / 'real.jpg'), 'label': 0,
                    'group': 'aigi:camera'}
            fake = {'path': str(local / 'fake.jpg'), 'label': 1,
                    'group': 'aigi:generator'}
            split = {'train': [old['train'][0], fake],
                     'validation_selection': [real],
                     'validation_calibration': [],
                     'test_reserved': old['test']}
            sources = [(manifest, local, 'aigi')]
            self.assertEqual(audit.verify_partitions(split, old, sources)['train'], 2)
            swapped = {**split, 'train': [old['train'][0], real],
                       'validation_selection': [fake]}
            with self.assertRaisesRegex(ValueError, 'atravessa divisões'):
                audit.verify_partitions(swapped, old, sources)
            changed_test = {**split, 'test_reserved': []}
            with self.assertRaisesRegex(ValueError, 'reservado'):
                audit.verify_partitions(changed_test, old, sources)

    def test_segmentos_usam_denominador_da_classe_e_arquivo_de_erros_vazio(self):
        rows = [{'dataset': 'AIGI', 'label': 0, 'error_type': 'FP'},
                {'dataset': 'AIGI', 'label': 0, 'error_type': ''},
                {'dataset': 'AIGI', 'label': 1, 'error_type': 'FN'}]
        segments = audit.segment_table(rows, ('dataset',), errors=True)
        self.assertEqual([(x['label'], x['n'], x['errors']) for x in segments],
                         [(0, 2, 1), (1, 1, 1)])
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'errors.csv'
            audit.write_csv(path, [], fieldnames=('dataset', 'label'))
            with path.open(newline='', encoding='utf-8') as handle:
                self.assertEqual(list(csv.DictReader(handle)), [])

    def test_metadados_nao_abrem_calibracao_ou_teste(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'train.jpg'
            Image.new('RGB', (300, 400), (80, 90, 100)).save(path, format='JPEG')
            row = {'path': str(path), 'label': 0, 'group': 'real'}
            split = {'train': [row], 'validation_selection': [row],
                     'validation_calibration': [{'path': 'ausente-calibracao.jpg'}],
                     'test_reserved': [{'path': 'ausente-teste.jpg'}]}
            index = {audit.key(row): {'dataset': 'Teste',
                                      'original_format': 'JPEG',
                                      'real_source': 'teste',
                                      'original_short_side': 300}}
            records = audit.collect_metadata(split, index)
            self.assertEqual(len(records), 2)
            self.assertTrue(all(item['processed_size_band'] == '256-511'
                                for item in records))
            audit.write_csv(Path(temp) / 'metadata.csv', records)


if __name__ == '__main__':
    unittest.main()
