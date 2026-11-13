"""Testes sem GPU do protocolo de avaliação externa congelada."""

import csv
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / 'Notebooks/fase_3/fase3_4/avaliar_externo_fase3_4_v2.py'
spec = importlib.util.spec_from_file_location('avaliar_externo_v2', SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ExternalEvaluationPreflightTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.external = self.root / 'validacao_fase3' / 'teste_externo'
        self.external.mkdir(parents=True)
        self.rows = []
        for index in range(20):
            name = f'imagem_{index:02d}.jpg'
            (self.external / name).write_bytes(f'externa-{index}'.encode())
            self.rows.append({'arquivo': name, 'label_real': '0' if index < 8 else '1'})
        self.write_labels()

    def write_labels(self):
        with (self.external / 'rotulos.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=('arquivo', 'label_real'))
            writer.writeheader()
            writer.writerows(self.rows)

    def test_valid_external_set_has_twenty_unique_images(self):
        _, rows = module.load_external(self.root)
        self.assertEqual(len(rows), 20)
        self.assertEqual(len({row['sha256'] for row in rows}), 20)
        self.assertEqual(sum(row['label'] for row in rows), 12)

    def test_rejects_duplicate_content(self):
        (self.external / 'imagem_19.jpg').write_bytes((self.external / 'imagem_18.jpg').read_bytes())
        with self.assertRaisesRegex(ValueError, 'duplicado'):
            module.load_external(self.root)

    def test_rejects_missing_file_and_changed_labels(self):
        (self.external / 'imagem_19.jpg').unlink()
        with self.assertRaises(FileNotFoundError):
            module.load_external(self.root)
        (self.external / 'imagem_19.jpg').write_bytes(b'externa-19')
        self.rows[19]['label_real'] = '0'
        self.write_labels()
        with self.assertRaisesRegex(ValueError, 'Distribuição'):
            module.load_external(self.root)

    def test_rejects_overlap_with_registered_partitions(self):
        _, external_rows = module.load_external(self.root)
        manifest_dir = self.root / 'dados_fase3_4'
        manifest_dir.mkdir()
        manifest = manifest_dir / 'manifest.json'
        manifest.write_text('{}', encoding='utf-8')
        run_dir = self.root / 'modelos' / module.RUN_NAME
        run_dir.mkdir(parents=True)
        split = {'manifest_sha256': hashlib.sha256(b'{}').hexdigest()}
        for index, key in enumerate(('train', 'validation_selection',
                                     'validation_calibration', 'test_reserved')):
            path = self.root / f'openfake_{index}.jpg'
            path.write_bytes(f'openfake-{index}'.encode())
            split[key] = [{'path': str(path)}]
        (run_dir / 'split_manifest.json').write_text(json.dumps(split), encoding='utf-8')
        module.verify_no_openfake_overlap(self.root, external_rows,
                                          progress_every=4, expected_count=4)
        (self.root / 'openfake_3.jpg').write_bytes((self.external / 'imagem_00.jpg').read_bytes())
        with self.assertRaisesRegex(ValueError, 'externa encontrada'):
            module.verify_no_openfake_overlap(self.root, external_rows,
                                              progress_every=4, expected_count=4)


if __name__ == '__main__':
    unittest.main()
