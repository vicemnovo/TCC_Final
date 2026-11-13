"""Regras locais do piloto 3.9; não faz downloads."""

import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experimentos'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'Notebooks/fase_3/fase3_9'))
from fase3_9_piloto_aigi import new_candidates


class FakeParquet:
    schema_arrow = SimpleNamespace(names=['uid', 'generator', 'height', 'width', 'image'])

    def __init__(self, path):
        self.path = path

    def iter_batches(self, **_):
        rows = [
            {'uid': 'USED0001', 'generator': 'FLUX.1_dev', 'height': 512, 'width': 512},
            {'uid': 'NEW00001', 'generator': 'FLUX.1_dev', 'height': 512, 'width': 512},
            {'uid': 'TINY0001', 'generator': 'FLUX.1_dev', 'height': 128, 'width': 512},
        ]
        yield SimpleNamespace(to_pylist=lambda: rows)


class PilotTests(unittest.TestCase):
    def test_existing_uid_and_small_image_excluded(self):
        real = {uid: {'source_dataset': 'MSCOCO',
                      'image_path': f'COCO_val2014_{uid}.jpg'}
                for uid in ('USED0001', 'NEW00001', 'TINY0001')}
        with patch.dict(sys.modules, {'pyarrow': SimpleNamespace(parquet=SimpleNamespace(
                ParquetFile=FakeParquet)),
                'pyarrow.parquet': SimpleNamespace(ParquetFile=FakeParquet)}):
            rows = new_candidates({'FLUX.1_dev': [Path('shard.parquet')]},
                                  real, {'USED0001'})
        self.assertEqual([row['uid'] for row in rows['FLUX.1_dev']], ['NEW00001'])

    def test_duplicate_uid_rejected(self):
        real = {'NEW00001': {'source_dataset': 'MSCOCO',
                            'image_path': 'COCO_val2014_NEW00001.jpg'}}
        with patch.dict(sys.modules, {'pyarrow': SimpleNamespace(parquet=SimpleNamespace(
                ParquetFile=FakeParquet)),
                'pyarrow.parquet': SimpleNamespace(ParquetFile=FakeParquet)}):
            with self.assertRaisesRegex(ValueError, 'UID repetido'):
                new_candidates({'FLUX.1_dev': [Path('a.parquet'), Path('b.parquet')]},
                               real, set())


if __name__ == '__main__':
    unittest.main()
