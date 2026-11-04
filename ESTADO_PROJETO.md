# Estado do projeto — Fase 3.8 pronta para treino

Atualizado em 03/11/2026 com as saídas revisadas dos notebooks e os resultados
informados pelo grupo. O Drive não foi montado nesta revisão; a existência
remota dos artefatos foi verificada pelas saídas do Colab, não por leitura
direta nesta máquina. O registro detalhado anterior está em
[`docs/historico/ESTADO_PROJETO_ate_3_6.md`](docs/historico/ESTADO_PROJETO_ate_3_6.md).

## Onde paramos

- **Branch:** `fase3-ajuste-fino`, acompanhando `origin/fase3-ajuste-fino` no
  início desta revisão. Em outro computador, confirme o commit com
  `git status -sb` e `git rev-parse HEAD` antes de abrir o Colab.
- **Objetivo:** detector de imagens reais e sintéticas com pelo menos 85% de
  acurácia em teste externo independente ainda não observado. Essa meta não
  foi demonstrada. Vídeos e aplicação web/API permanecem para etapa futura.
- **Fase atual:** 3.8, ajuste fino com pares AIGI. O preparo foi concluído:
  **480 pares/960 imagens**, 80 pares por gerador, com 640 imagens adicionais
  no treino, 160 na seleção e 160 na calibração. Os dados estão em
  `/content/drive/MyDrive/TCC_Final/dados_fase3_8/aigi_pares_256_v1/`.
- **Cópia local:** a célula `# 8` confirmou as 960 imagens AIGI em
  `/content/aigi_pares_256_v1/`, além das cópias locais 3.5 e 3.7.
- **Smoke test:** a célula `# 9` terminou com `Smoke OK`, GPU, arquitetura,
  checkpoint inicial da **Fase 3.3**, forward e gravação. Conferiu por hash
  6.152 imagens de treino, 1.420 de seleção, 1.420 de calibração e 1.947 do
  teste OpenFake reservado. O teste reservado não foi inferido.
- **Treino 3.8:** a célula `# 10` ainda não tem saída no notebook versionado.
  Nenhuma época, métrica interna ou resultado Mirage da 3.8 está verificado
  aqui. Não apresentar o smoke como treino concluído.

## Resultados anteriores comparáveis

| Modelo | Mirage 2000: acurácia | AUC | Estado |
| --- | ---: | ---: | --- |
| Fase 3.5 | 60,45% | 0,679972 | Treino e avaliação concluídos |
| Fase 3.6 | 57,20% | 0,6303885 | Treino e avaliação concluídos |
| Fase 3.7 | 62,30% | 0,6908315 | Treino e avaliação concluídos |
| Fase 3.8 | — | — | Preparo e smoke concluídos; treino pendente |

As métricas acima vieram das saídas do Colab enviadas pelo grupo. O Mirage
contém 1.000 imagens reais e 1.000 sintéticas fixas, mas já foi observado.
Seu uso em novas fases é comparativo; não usar suas respostas para treinar,
calibrar ou escolher pesos, dados ou hiperparâmetros. A avaliação independente
final exigirá outro conjunto reservado antes de ser visto. O benchmark
RAISE/MNW e as 20 imagens externas antigas também são históricos.

## Como continuar a 3.8

Abra [`Notebooks/fase_3/rodar_fase3_8_colab.ipynb`](Notebooks/fase_3/rodar_fase3_8_colab.ipynb)
no VS Code com GPU do Colab. A numeração abaixo é a que aparece nos comentários
`# N.` das células de código; a primeira célula Markdown não entra na conta.

1. **Sessão já conectada, com `# 8` e `# 9` concluídas:** execute manualmente
   **`# 10. TREINO MANUAL`**. O modo `auto` inicia ou retoma de `last.pt`.
2. **Nova sessão:** execute **`# 1` a `# 5`** para GPU, dependências, Drive,
   credencial GitHub e snapshot do código. Execute **`# 8. COPIA LOCAL`** para
   refazer `/content` e depois **`# 10`**. O preparo `# 7` já concluiu; não
   refaça a seleção. O smoke `# 9` já foi salvo e não precisa ser repetido.
3. Ao terminar, execute **`# 11. LEITURA INTERNA`** e confira
   `training_complete.json`, `internal_metrics.json`, melhor época, AUC de
   seleção, temperatura e limiar. Treino concluído ainda não significa
   avaliação externa concluída.
4. Execute **`# 12` a `# 15`** em ordem: cópia local do Mirage, congelamento
   de checkpoint/temperatura/limiar, avaliação comparativa e leitura do
   resultado. Envie as saídas de `# 11` e `# 15` para registrar a 3.8.

Os checkpoints novos ficam somente em
`/content/drive/MyDrive/TCC_Final/modelos/fase3_8_aigi_pares256_v1/`;
resultados comparativos em
`/content/drive/MyDrive/TCC_Final/resultados/benchmark_externo_mirage_2000_v1/fase3_8_aigi_pares256_v1/`.
Os modelos 3.3–3.7 continuam em diretórios próprios. Nunca substituir seus
checkpoints, métricas ou manifestos.

## Limitações e próximo trabalho

- A auditoria do smoke mostrou associação entre classe e resolução ainda
  presente nos dados Community Forensics: 1.936 reais tinham lado curto
  256–511 px, enquanto 1.744 sintéticas tinham 512 px ou mais no treino.
  Os 320 pares AIGI de treino foram normalizados para JPEG e lado curto
  256 px nas duas classes; isso não elimina o viés dos dados anteriores.
- A preparação AIGI exigiu dois reparos antes de concluir: ampliação
  documentada do plano com três shards e substituição de sintéticas cujos
  bytes contradiziam os metadados de tamanho. Plano, extensão, estado e hashes
  foram preservados no Drive.
- Após a avaliação 3.8, escolher a próxima hipótese **apenas com dados
  internos**: diversidade de novos datasets, qualidade dos rótulos e controle
  de atalhos de tamanho/formato têm prioridade. O teste Mirage relata
  generalização observada e não determina ajustes da próxima versão.

Para entender a arquitetura e as métricas, leia
[`docs/GUIA_METODOLOGIA_TCC.md`](docs/GUIA_METODOLOGIA_TCC.md). Para detalhes
de preparo, retomada e avaliação da 3.8, leia
[`docs/README_fase3_8_execucao.md`](docs/README_fase3_8_execucao.md).
