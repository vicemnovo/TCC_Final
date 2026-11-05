"""A linguagem deve conservar probabilidades e o limiar real da fase."""

from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experimentos'))
from linguagem_predicao import descrever_predicao


class LinguagemPredicaoTest(unittest.TestCase):
    def test_frase_nao_confunde_maior_probabilidade_com_decisao_por_limiar(self):
        # Na Fase 3.8, p=0,40 é menor que 50%, mas supera o limiar 0,31.
        result = descrever_predicao(0.40, 0.31)
        self.assertEqual(result['classificacao'], 'sintética')
        self.assertAlmostEqual(result['p_real'], 0.60)
        self.assertIn('próximas', result['frase'])
        self.assertIn('40.0%', result['frase'])
        self.assertIn('31.0%', result['frase'])

    def test_extremos_e_sem_calibracao(self):
        self.assertIn('chance estimada alta de ser real',
                      descrever_predicao(0.05, 0.47)['frase'])
        self.assertIn('chance estimada alta de ser sintética',
                      descrever_predicao(0.82, 0.47)['frase'])
        self.assertIsNone(descrever_predicao(0.82, None)['classificacao'])

    def test_rejeita_probabilidade_ou_limiar_invalidos(self):
        for probability, threshold in ((float('nan'), 0.47), (1.2, 0.47),
                                       (0.4, 0.0), (0.4, float('inf'))):
            with self.subTest(probability=probability, threshold=threshold):
                with self.assertRaises(ValueError):
                    descrever_predicao(probability, threshold)


if __name__ == '__main__':
    unittest.main()
