"""Checks locais do preparo 3.8; nao acessam Drive nem iniciam treino."""

import io
from pathlib import Path
import sys
import unittest

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experimentos'))
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


if __name__ == '__main__':
    unittest.main()
