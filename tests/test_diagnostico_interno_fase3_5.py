"""Verifica o diagnóstico interno sem carregar modelo ou usar GPU."""

from pathlib import Path
import sys
import tempfile
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experimentos'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'Notebooks/fase_3/fase3_5'))
import diagnostico_interno_fase3_5 as diagnostic


class DiagnosticoInternoTest(unittest.TestCase):
    def test_inferencia_fica_restrita_a_selecao(self):
        self.assertEqual(diagnostic.DIAGNOSTIC_PARTITIONS, ('validation_selection',))
        self.assertIn('validation_calibration', diagnostic.REGISTERED_PARTITIONS)

    def test_resumo_de_erros_com_limiar_congelado(self):
        rows = [{'label': 0, 'group': 'real'}, {'label': 0, 'group': 'real'},
                {'label': 1, 'group': 'fake'}, {'label': 1, 'group': 'fake'}]
        summary = diagnostic.summarize(rows, [0.1, 0.8, 0.2, 0.9], 0.445)
        self.assertEqual((summary['tn'], summary['fp'], summary['fn'], summary['tp']),
                         (1, 1, 1, 1))
        self.assertEqual(summary['balanced_accuracy'], 0.5)
        report = diagnostic.group_table('validation_selection', rows,
                                        [0.1, 0.8, 0.2, 0.9], 0.445)
        self.assertEqual([(item['label'], item['errors']) for item in report],
                         [(0, 1), (1, 1)])

    def test_particoes_devem_corresponder_ao_manifesto_sem_inferir_teste(self):
        with tempfile.TemporaryDirectory() as temporary:
            local = Path(temporary)
            old = {'train': [{'path': 'old.jpg', 'label': 0, 'group': 'old'}],
                   'test': [{'path': 'reserved.jpg', 'label': 1, 'group': 'reserved'}]}
            new = {'rows': [{'image': 'images/1/new.jpg', 'label': 1, 'group': 'gen'}]}
            new_row = {'path': str(local / 'images/1/new.jpg'),
                       'label': 1, 'group': 'cfsmall:gen'}
            split = {'train': old['train'], 'validation_selection': [new_row],
                     'validation_calibration': [], 'test_reserved': old['test']}
            diagnostic.expected_partitions(split, old, new, local)
            changed = {**split, 'validation_selection': [
                {**new_row, 'group': 'different'}]}
            with self.assertRaisesRegex(ValueError, 'Partições'):
                diagnostic.expected_partitions(changed, old, new, local)
            changed = {**split, 'test_reserved': []}
            with self.assertRaisesRegex(ValueError, 'reservado'):
                diagnostic.expected_partitions(changed, old, new, local)


if __name__ == '__main__':
    unittest.main()
