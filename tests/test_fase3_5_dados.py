"""Testes locais sem rede nem GPU para a seleção interna da Fase 3.5."""

from collections import Counter, defaultdict
import io
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    'fase3_5_dados', ROOT / 'experimentos/fase3_5_dados.py')
data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data)


class Fase35DataTests(unittest.TestCase):
    def fixture_groups(self):
        groups = defaultdict(list)
        for label in (0, 1):
            for group in ('a', 'b', 'c', 'd', 'e'):
                for index in range(30):
                    groups[label, group].append({
                        'shard': f'data/HFCF_small_{label}.parquet',
                        'index': 1000 * label + 100 * ord(group) + index,
                        'label': label, 'group': group, 'image_name': f'{group}_{index}.png',
                    })
        return groups

    def test_selecao_deterministica_e_grupos_separados(self):
        groups = self.fixture_groups()
        first = data.select_rows(groups, 100, 25)
        self.assertEqual(first, data.select_rows(groups, 100, 25))
        self.assertEqual(Counter(row['label'] for row in first), {0: 100, 1: 100})
        for label in (0, 1):
            assigned = defaultdict(set)
            for row in first:
                if row['label'] == label:
                    assigned[row['group']].add(row['subset'])
            self.assertTrue(all(len(subsets) == 1 for subsets in assigned.values()))
            self.assertEqual({next(iter(value)) for value in assigned.values()}, set(data.SPLITS))

    def test_amostra_insuficiente_e_recusada(self):
        groups = self.fixture_groups()
        groups.pop((1, 'c'))
        groups.pop((1, 'd'))
        groups.pop((1, 'e'))
        with self.assertRaises(ValueError):
            data.select_rows(groups, 100, 25)

    def test_stage_confere_hash_e_preserva_manifesto(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'dados_fase3_5' / data.NAME
            image = source / 'images/0/example.jpg'
            image.parent.mkdir(parents=True)
            image.write_bytes(b'fixture')
            manifest = {'rows': [{'image': 'images/0/example.jpg',
                                  'image_sha256': data.sha(image)}]}
            (source / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
            class Args:
                pass
            args = Args()
            args.root, args.local_dir = root, root / 'content'
            data.stage(args)
            self.assertEqual((args.local_dir / 'images/0/example.jpg').read_bytes(), b'fixture')
            (args.local_dir / 'images/0/example.jpg').write_bytes(b'alterado')
            with self.assertRaises(ValueError):
                data.stage(args)

    def test_jpeg_gerado_suporta_recorte_224(self):
        raw = io.BytesIO()
        Image.new('RGB', (120, 180), (20, 30, 40)).save(raw, format='PNG')
        encoded, size = data.encode_jpeg(raw.getvalue())
        self.assertGreaterEqual(min(size), 224)
        with Image.open(io.BytesIO(encoded)) as image:
            self.assertEqual(image.format, 'JPEG')


if __name__ == '__main__':
    unittest.main()
