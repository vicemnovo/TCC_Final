"""Frases padronizadas para apresentar predições individuais futuras.

A frase descreve a pontuação do modelo. A classe operacional continua sendo
definida pelo limiar congelado da fase; as faixas de texto não mudam a decisão.
Resultados históricos com hashes registrados não são reescritos aqui.
"""

from math import isfinite


def descrever_predicao(p_sintetica: float, limiar: float | None) -> dict:
    """Devolve duas probabilidades, classe operacional e frase pronta.

    `limiar=None` é permitido para uma demonstração sem calibração, mas não
    produz classe binária. Números são estimativas, nunca prova de origem.
    """
    p_sintetica = float(p_sintetica)
    if not isfinite(p_sintetica) or not 0.0 <= p_sintetica <= 1.0:
        raise ValueError('Probabilidade sintética deve estar entre 0 e 1')
    if limiar is not None:
        limiar = float(limiar)
        if not isfinite(limiar) or not 0.0 < limiar < 1.0:
            raise ValueError('Limiar interno deve estar entre 0 e 1')

    p_real = 1.0 - p_sintetica
    if p_sintetica < 0.20:
        trecho = 'há uma chance estimada alta de ser real'
    elif p_sintetica < 0.40:
        trecho = 'há mais chance estimada de ser real'
    elif p_sintetica < 0.60:
        trecho = 'as chances estimadas de ser real ou sintética estão próximas'
    elif p_sintetica < 0.80:
        trecho = 'há mais chance estimada de ser sintética'
    else:
        trecho = 'há uma chance estimada alta de ser sintética'

    classificacao = None if limiar is None else (
        'sintética' if p_sintetica >= limiar else 'real')
    frase = (f'Segundo o modelo, {trecho}. '
             f'Probabilidades estimadas: real {p_real:.1%}; '
             f'sintética {p_sintetica:.1%}.')
    if classificacao is None:
        frase += ' Sem calibração persistida: não há classificação binária confiável.'
    else:
        frase += (f' Pelo limiar interno congelado de {limiar:.1%}, '
                  f'a classificação operacional é {classificacao}.')
    return {
        'p_real': p_real,
        'p_sintetica': p_sintetica,
        'classificacao': classificacao,
        'limiar': limiar,
        'frase': frase,
        'nota': 'Estimativa do modelo; não comprova a origem nem a veracidade da cena.',
    }
