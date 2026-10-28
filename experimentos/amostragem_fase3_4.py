"""Pesos de amostragem da Fase 3.4 v2, sem dependência de GPU."""

from collections import Counter
from math import sqrt


def balanced_sqrt_group_weights(rows):
    """Equilibra classes; massa de cada grupo cresce como raiz do seu tamanho.

    A v1 usava massa igual por grupo, independentemente de ter uma ou cem
    imagens. A raiz quadrada reduz repetições dos grupos minúsculos sem fazer
    os geradores grandes dominarem como em uma amostragem por imagem.
    """
    counts = Counter((int(row["label"]), str(row["group"])) for row in rows)
    labels = {label for label, _ in counts}
    if labels != {0, 1}:
        raise ValueError("O treino precisa conter as classes real=0 e fake=1")

    class_mass = {
        label: sum(sqrt(n) for (lab, _), n in counts.items() if lab == label)
        for label in (0, 1)
    }
    return [
        0.5 / (sqrt(counts[(int(row["label"]), str(row["group"]))])
               * class_mass[int(row["label"])])
        for row in rows
    ]
