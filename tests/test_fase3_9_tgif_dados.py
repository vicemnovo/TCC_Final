"""Testes locais da selecao pareada TGIF, sem baixar arquivos grandes."""

from collections import Counter
import csv
import io
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experimentos'))
import fase3_9_tgif_dados as dados


def png(color=(25, 50, 75)):
    buffer = io.BytesIO()
    Image.new('RGB', (512, 512), color).save(buffer, format='PNG')
    return buffer.getvalue()


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


class TgifDataTests(unittest.TestCase):
    def test_plan_keeps_duplicate_category_ids_together(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ids = [str(i) for i in range(1558)]
            originals = [{'image_id': image_id, 'category': f'cat{i % 53}'}
                         for i, image_id in enumerate(ids)]
            originals += [{'image_id': ids[i], 'category': f'other{i % 53}'}
                          for i in range(882)]
            fake = [{'image_id': row['image_id'], 'mask_variation': mask,
                     'gen_variation_id': str(variation)}
                    for row in originals for mask in ('bbox', 'segm')
                    for variation in range(3)]
            paths = {name: root / name for name in dados.METADATA_FILES}
            write_csv(paths['orig_training.csv'], originals)
            write_csv(paths['sd2_training.csv'], fake)
            for name in ('orig_validation.csv', 'orig_testing.csv'):
                write_csv(paths[name], [{'image_id': '999999', 'category': 'held'}])
            plan = dados.make_plan(paths)
            self.assertEqual(len(plan['rows']), 768)
            self.assertEqual(len({row['image_id'] for row in plan['rows']}), 768)
            self.assertEqual(Counter(row['subset'] for row in plan['rows']),
                             {'train': 512, 'validation_selection': 128,
                              'validation_calibration': 128})
            dados.validate_plan(plan, paths)

    def test_selected_archive_never_extracts_unselected_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive_path = root / 'original.tar.gz'
            content = png()
            with tarfile.open(archive_path, 'w:gz') as archive:
                for name in ('training/truck/12_orig_512.png',
                             'training/cat/12_orig_512.png',
                             'training/truck/13_orig_512.png'):
                    raw = png((90, 20, 10)) if '/cat/' in name else content
                    member = tarfile.TarInfo(name)
                    member.size = len(raw)
                    archive.addfile(member, io.BytesIO(raw))
            plan = {'rows': [{'category': 'truck',
                              'original_member': '12_orig_512.png'}]}
            with patch.object(dados, 'COUNT', 1):
                selected = dados.read_selected(archive_path, plan, 'original')
            self.assertEqual(list(selected), ['training/truck/12_orig_512.png'])
            self.assertEqual(selected['training/truck/12_orig_512.png'], content)
            self.assertFalse((root / 'training').exists())


if __name__ == '__main__':
    unittest.main()
