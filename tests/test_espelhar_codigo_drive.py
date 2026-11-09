"""Confere o espelho sem montar Drive nem acessar o Colab."""

import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experimentos'))
from espelhar_codigo_drive import mirror


COMMIT = 'a' * 40


class EspelhoCodigoDriveTest(unittest.TestCase):
    def test_preserva_pastas_arquivos_anteriores_e_dados(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'snapshot'
            drive = root / 'drive' / 'TCC_Final'
            notebook = Path('Notebooks/fase_3/novo.ipynb')
            script = Path('experimentos/novo.py')
            (source / notebook).parent.mkdir(parents=True)
            (source / script).parent.mkdir(parents=True)
            (source / notebook).write_bytes(b'{"cells": []}')
            (source / script).write_bytes(b'print("versao nova")\n')
            (drive / notebook).parent.mkdir(parents=True)
            (drive / notebook).write_bytes(b'{"cells": ["versao anterior"]}')
            (drive / 'modelos').mkdir(parents=True)
            (drive / 'modelos' / 'last.pt').write_bytes(b'checkpoint preservado')

            result = mirror(source, drive, COMMIT)
            self.assertEqual(result['file_count'], 2)
            self.assertEqual(result['copied'], 2)
            self.assertEqual(result['backed_up'], 1)
            self.assertEqual((drive / notebook).read_bytes(), b'{"cells": []}')
            self.assertEqual((drive / script).read_bytes(), b'print("versao nova")\n')
            previous = b'{"cells": ["versao anterior"]}'
            backup = drive / 'codigo_historico' / hashlib.sha256(previous).hexdigest() / notebook
            self.assertEqual(backup.read_bytes(), previous)
            self.assertEqual((drive / 'modelos' / 'last.pt').read_bytes(),
                             b'checkpoint preservado')
            manifest = json.loads(Path(result['manifest']).read_text(encoding='utf-8'))
            self.assertEqual([item['path'] for item in manifest['files']],
                             [notebook.as_posix(), script.as_posix()])

            repeated = mirror(source, drive, COMMIT)
            self.assertEqual(repeated['copied'], 0)
            self.assertEqual(repeated['unchanged'], 2)
            self.assertEqual(repeated['backed_up'], 0)

    def test_nao_mistura_codigo_com_pastas_reservadas(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'snapshot'
            drive = root / 'drive'
            (source / 'Notebooks').mkdir(parents=True)
            (source / 'Notebooks' / 'demo.ipynb').write_bytes(b'{}')
            (source / 'modelos').mkdir()
            (source / 'modelos' / 'helper.py').write_bytes(b'pass')
            drive.mkdir()
            with self.assertRaisesRegex(ValueError, 'pasta reservada'):
                mirror(source, drive, COMMIT)
            self.assertFalse((drive / 'Notebooks').exists())

    def test_manifesto_de_mesmo_commit_divergente_interrompe_antes_de_copiar(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'snapshot'
            drive = root / 'drive'
            (source / 'Notebooks').mkdir(parents=True)
            (source / 'experimentos').mkdir()
            (source / 'Notebooks' / 'demo.ipynb').write_bytes(b'{}')
            (source / 'experimentos' / 'demo.py').write_bytes(b'pass')
            (drive / 'codigo_sync').mkdir(parents=True)
            (drive / 'codigo_sync' / f'{COMMIT}.json').write_bytes(b'outro manifesto')
            with self.assertRaisesRegex(IOError, 'divergente'):
                mirror(source, drive, COMMIT)
            self.assertFalse((drive / 'Notebooks').exists())


if __name__ == '__main__':
    unittest.main()
