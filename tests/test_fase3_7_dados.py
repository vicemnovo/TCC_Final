"""Verifica isolamento de geradores e consistência da seleção 3.7."""

from collections import defaultdict
import unittest

from Notebooks.fase_3.fase3_7.fase3_7_dados import select_rows


class SelecaoFase37(unittest.TestCase):
    def test_gerador_antigo_permanece_na_mesma_divisao(self):
        groups = defaultdict(list)
        for index in range(200):
            groups[0, 'n/a'].append({'shard': 'real', 'index': index,
                                      'label': 0, 'group': 'n/a', 'image_name': str(index)})
        for group in ('adm', 'biggan', 'glide', 'midjourney', 'sd14', 'sd15'):
            for index in range(50):
                groups[1, group].append({'shard': 'fake', 'index': index,
                                          'label': 1, 'group': group,
                                          'image_name': f'{group}-{index}'})
        prior = {(1, 'adm'): 'validation_selection',
                 (1, 'biggan'): 'validation_calibration'}
        rows = select_rows(groups, per_class=160, per_group=40, prior_groups=prior)
        for group, expected in prior.items():
            self.assertEqual({row['subset'] for row in rows
                              if (row['label'], row['group']) == group}, {expected})
        for subset in ('train', 'validation_selection', 'validation_calibration'):
            self.assertTrue({row['label'] for row in rows if row['subset'] == subset} == {0, 1})


if __name__ == '__main__':
    unittest.main()
