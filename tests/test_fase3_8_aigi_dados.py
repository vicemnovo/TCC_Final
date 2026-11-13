"""Checks locais do preparo 3.8; nao acessam Drive nem iniciam treino."""

import io
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experimentos'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'Notebooks/fase_3/fase3_8'))
import fase3_8_aigi_dados as aigi
from fase3_8_aigi_dados import normalize, real_url


class AigiDataTests(unittest.TestCase):
    def test_normalize_uses_same_jpeg_256_for_both_classes(self):
        for source_format in ('PNG', 'JPEG'):
            with self.subTest(source_format=source_format):
                buffer = io.BytesIO()
                Image.new('RGB', (600, 400), (80, 120, 160)).save(
                    buffer, format=source_format)
                encoded, original, size = normalize(buffer.getvalue())
                self.assertEqual(original['format'], source_format)
                self.assertEqual(min(size), 256)
                with Image.open(io.BytesIO(encoded)) as result:
                    self.assertEqual(result.format, 'JPEG')
                    self.assertEqual(min(result.size), 256)

    def test_no_upscale_of_small_real_image(self):
        buffer = io.BytesIO()
        Image.new('RGB', (400, 255)).save(buffer, format='JPEG')
        with self.assertRaisesRegex(ValueError, 'menor'):
            normalize(buffer.getvalue())

    def test_real_urls_are_restricted_to_selected_sources(self):
        self.assertEqual(
            real_url({'source_dataset': 'MSCOCO',
                      'image_path': 'COCO_val2014_000000000042.jpg'}),
            'https://s3.amazonaws.com/images.cocodataset.org/val2014/'
            'COCO_val2014_000000000042.jpg')
        self.assertIsNone(real_url({'source_dataset': 'SA-1B',
                                    'image_path': 'https://example.org/a.jpg'}))
        self.assertIsNone(real_url({'source_dataset': 'CC3M',
                                    'image_path': 'http://example.org/a.jpg'}))

    def test_fake_with_false_size_metadata_is_rejected_before_manifest(self):
        buffer = io.BytesIO()
        Image.new('RGB', (128, 128)).save(buffer, format='PNG')
        shard = Path(aigi.SHARDS['FLUX.1_dev']).name
        item = {'uid': 'abcd1234', 'shard': shard, 'index': 0,
                'real_image_sha256': 'a' * 64,
                'real_original_sha256': 'b' * 64,
                'real_image': 'images/0/abcd1234.jpg',
                'real_original_format': 'JPEG',
                'real_original_size': [512, 512],
                'processed_size': [256, 256],
                'subset': 'train', 'real_source': 'MSCOCO'}

        class Batch:
            def __init__(self, image):
                self.image = image

            def to_pylist(self):
                return [{'image': {'bytes': self.image}}]

        class ParquetFile:
            def __init__(self, path):
                self.path = Path(path)

            def iter_batches(self, **_):
                return [Batch(buffer.getvalue())] if self.path.name == shard else []

        parquet = types.ModuleType('pyarrow.parquet')
        parquet.ParquetFile = ParquetFile
        pyarrow = types.ModuleType('pyarrow')
        pyarrow.parquet = parquet
        files = {name: Path(name) for name in
                 (*aigi.SHARDS.values(), *aigi.EXTRA_SHARDS.values())}
        with tempfile.TemporaryDirectory() as directory, patch.dict(
                sys.modules, {'pyarrow': pyarrow, 'pyarrow.parquet': parquet}):
            rows, invalid = aigi.materialize_selected(
                Path(directory), files, {'FLUX.1_dev': [item]}, set())
        self.assertEqual(rows, [])
        self.assertEqual(invalid, [('FLUX.1_dev', 'abcd1234', 'ValueError')])


if __name__ == '__main__':
    unittest.main()
