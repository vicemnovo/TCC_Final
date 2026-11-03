"""Verificações locais do piloto AIGI sem baixar imagens."""

import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experimentos.fase3_8_aigi_piloto import public_https_url, real_rows


class AigiPilotTests(unittest.TestCase):
    def test_url_real_rejeita_destino_local_e_http(self):
        with self.assertRaises(ValueError):
            public_https_url('http://example.org/photo.jpg')
        with patch('socket.getaddrinfo', return_value=[
                (2, 1, 6, '', ('127.0.0.1', 443))]):
            with self.assertRaises(ValueError):
                public_https_url('https://example.org/photo.jpg')
        with patch('socket.getaddrinfo', return_value=[
                (2, 1, 6, '', ('8.8.8.8', 443))]):
            self.assertEqual(public_https_url('https://example.org/photo.jpg'),
                             'https://example.org/photo.jpg')

    def test_csv_real_exige_4000_uids_distintos(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'real.csv'
            with path.open('w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=(
                    'uid', 'source_dataset', 'image_path'))
                writer.writeheader()
                for number in range(4000):
                    writer.writerow({'uid': f'uid{number:04d}',
                                     'source_dataset': 'CC3M',
                                     'image_path': 'https://example.org/photo.jpg'})
            rows, sources = real_rows(path)
            self.assertEqual(len(rows), 4000)
            self.assertEqual(sources['CC3M'], 4000)
            with path.open('a', newline='', encoding='utf-8') as stream:
                csv.writer(stream).writerow(
                    ['uid0000', 'CC3M', 'https://example.org/other.jpg'])
            with self.assertRaisesRegex(ValueError, 'repetido'):
                real_rows(path)


if __name__ == '__main__':
    unittest.main()
