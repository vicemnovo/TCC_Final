"""Verificacoes locais do piloto TGIF, sem rede e sem arquivos grandes."""

import io
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experimentos'))
from fase3_9_tgif_piloto import member_id, scan_archive, source_url


def png():
    buffer = io.BytesIO()
    Image.new('RGB', (320, 256), (20, 40, 60)).save(buffer, format='PNG')
    return buffer.getvalue()


class TgifPilotTests(unittest.TestCase):
    def test_names_match_documented_pair(self):
        self.assertEqual(member_id('orig/193926_orig_512.png', 'original'), '193926')
        self.assertEqual(member_id(
            'sd2-fr/193926_mask_bbox.png_sd2-512_0.png', 'sd2_fr'), '193926')
        self.assertIsNone(member_id('other.jpg', 'sd2_fr'))
        with self.assertRaises(ValueError):
            source_url('../../outro.tar.gz')

    def test_tar_is_scanned_without_extracting_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            archive_path = Path(temporary) / 'sample.tar.gz'
            content = png()
            with tarfile.open(archive_path, 'w:gz') as archive:
                info = tarfile.TarInfo('orig/193926_orig.png')
                info.size = len(content)
                archive.addfile(info, io.BytesIO(content))
            report = scan_archive(archive_path, 'original')
            self.assertEqual(report['ids'], {'193926': 1})
            self.assertEqual(report['samples_decoded'], 1)
            self.assertEqual(report['sample_short_sides'], {'256': 1})
            self.assertFalse((Path(temporary) / 'orig').exists())


if __name__ == '__main__':
    unittest.main()
