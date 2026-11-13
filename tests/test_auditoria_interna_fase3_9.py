"""Confere isolamento e pareamento da auditoria sem Colab, GPU ou datasets."""

from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experimentos'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'Notebooks/fase_3/fase3_9'))
import auditoria_interna_fase3_9 as audit


def row(name, label, group):
    return {'path': name, 'label': label, 'group': group}


class AuditoriaInternaFase39Test(unittest.TestCase):
    def test_apenas_pares_tgif_podem_ser_acrescentados(self):
        before = {name: [row(f'{name}-antigo', 0, 'openfake')]
                  for name in audit.PARTITIONS}
        after = {name: list(rows) for name, rows in before.items()}
        for subset, count in (('train', 1024), ('validation_selection', 256),
                              ('validation_calibration', 256)):
            after[subset].extend(row(f'{subset}-{i}', i % 2,
                                     'tgif:coco' if i % 2 == 0 else 'tgif:sd2-fr')
                                 for i in range(count))
        growth = audit.compare_splits(before, after)
        self.assertEqual(growth['validation_selection']['tgif_added'], 256)
        moved = {name: list(rows) for name, rows in after.items()}
        moved['train'][0] = row('alterado', 0, 'openfake')
        with self.assertRaisesRegex(ValueError, 'além dos pares TGIF'):
            audit.compare_splits(before, moved)
        changed_test = {name: list(rows) for name, rows in after.items()}
        changed_test['test_reserved'].append(row('vazamento', 1, 'tgif:sd2-fr'))
        with self.assertRaisesRegex(ValueError, 'reservado mudou'):
            audit.compare_splits(before, changed_test)

    def test_pares_tgif_devem_ficar_na_mesma_divisao(self):
        rows = []
        for i in range(768):
            subset = ('train' if i < 512 else
                      'validation_selection' if i < 640 else 'validation_calibration')
            rows.extend({'image_id': str(i), 'label': label,
                         'subset': subset, 'category': 'cat'} for label in (0, 1))
        manifest = {'rows': rows}
        self.assertEqual(audit.verify_tgif_pairs(manifest)['train'], 512)
        changed = {'rows': [dict(item) for item in rows]}
        changed['rows'][1]['subset'] = 'validation_calibration'
        with self.assertRaisesRegex(ValueError, 'Par TGIF inválido'):
            audit.verify_tgif_pairs(changed)

    def test_comparacao_pareada_conta_todos_os_casos(self):
        rows = [row(str(i), label, 'teste') for i, label in enumerate((0, 0, 1, 1))]
        result = audit.compare_predictions(rows, [0, 0, 0, 1], [0, 1, 1, 0], range(4))
        self.assertEqual(result, {'both_correct': 1, 'only_phase3_8_correct': 2,
                                  'only_phase3_9_correct': 1, 'both_wrong': 0})

    def test_massa_do_sampler_fica_equilibrada_por_classe(self):
        rows = [row('r1', 0, 'tgif:coco'), row('r2', 0, 'tgif:coco'),
                row('f1', 1, 'tgif:sd2-fr'), row('f2', 1, 'outro')]
        index = {audit.key(item): {'dataset': 'TGIF' if item['group'].startswith('tgif:')
                                   else 'Outro'} for item in rows}
        mass = audit.sampler_mass(rows, index)
        by_class = {label: sum(item['sampling_mass'] for item in mass
                               if item['label'] == label) for label in (0, 1)}
        self.assertAlmostEqual(by_class[0], 0.5)
        self.assertAlmostEqual(by_class[1], 0.5)


if __name__ == '__main__':
    unittest.main()
