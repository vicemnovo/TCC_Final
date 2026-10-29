import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'benchmark_mirage_2000', ROOT / 'experimentos/benchmark_mirage_2000.py')
mirage = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mirage)


class MirageBenchmarkTest(unittest.TestCase):
    def rows(self):
        return [
            {'file_name': f'{domain}/{folder}/group/{index}.jpg',
             'is_real': folder, 'content_type': domain}
            for domain in mirage.DOMAINS
            for folder in ('0_real', '1_fake')
            for index in range(8)
        ]

    def test_selection_is_deterministic_balanced_and_excludes_anime(self):
        rows = self.rows() + [{'file_name': 'Anime/1_fake/g/0.jpg',
                               'is_real': '1_fake', 'content_type': 'Anime'}]
        with patch.object(mirage, 'PER_DOMAIN_CLASS', 3), patch.object(mirage, 'TOTAL', 24):
            a = mirage.select_rows(rows)
            b = mirage.select_rows(list(reversed(rows)))
        self.assertEqual(a, b)
        self.assertEqual(len(a), 24)
        self.assertEqual({(r['domain'], r['label']) for r in a},
                         {(d, c) for d in mirage.DOMAINS for c in (0, 1)})
        self.assertFalse(any('Anime' in r['source_file'] for r in a))

    def test_metadata_label_mismatch_fails_closed(self):
        rows = self.rows()
        rows[0]['is_real'] = '1_fake'
        with patch.object(mirage, 'PER_DOMAIN_CLASS', 3), patch.object(mirage, 'TOTAL', 24):
            with self.assertRaisesRegex(ValueError, 'Rótulo'):
                mirage.select_rows(rows)

    def test_stage_verifies_hash_and_refuses_changed_local_copy(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'dados_externos' / mirage.NAME
            local = root / 'content' / mirage.NAME
            records = []
            for label, folder in ((0, 'real'), (1, 'fake')):
                for index in range(2):
                    rel = f'images/Animal/{folder}/{index}.jpg'
                    path = source / rel
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(f'{folder}-{index}'.encode())
                    records.append({'label': label, 'image': rel,
                                    'image_sha256': mirage.digest(path)})
            (source / 'manifest.json').write_text(json.dumps({'rows': records}), encoding='utf-8')
            args = SimpleNamespace(root=root, local_dir=local)
            with patch.object(mirage, 'TOTAL', 4), patch.object(
                    mirage.shutil, 'disk_usage', return_value=SimpleNamespace(free=4_000_000_000)):
                mirage.stage(args)
                (local / records[0]['image']).write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError, 'Imagem local alterada'):
                    mirage.stage(args)


if __name__ == '__main__':
    unittest.main()
