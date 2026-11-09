# Estado do projeto — Fase 3.9 treinada e avaliada no Mirage

**Espelho de código no Drive preparado em 09/11/2026:** o notebook
`Notebooks/configuracao/sincronizar_codigo_drive_colab.ipynb` e o script
`experimentos/espelhar_codigo_drive.py` copiam `.ipynb` e `.py` da branch
`fase3-ajuste-fino` para as mesmas pastas em
`/content/drive/MyDrive/TCC_Final`. O Drive não foi montado nesta revisão:
o upload remoto ainda depende de executar as células 1–5 desse notebook
no Colab depois da publicação do código. Ver
`docs/SINCRONIZACAO_CODIGO_DRIVE.md`.

**Guia de retomada:** [`docs/GUIA_RETOMADA_FASE3_9.md`](docs/GUIA_RETOMADA_FASE3_9.md).
Ele registra as células exatas, o estado da demo independente e alternativas
para a próxima fase.
**Auditoria interna concluída:** o notebook
[`Notebooks/fase_3/auditar_interno_fase3_9_colab.ipynb`](Notebooks/fase_3/auditar_interno_fase3_9_colab.ipynb)
compara 3.8/3.9 apenas na seleção interna. Instruções em
[`docs/AUDITORIA_INTERNA_FASE3_9.md`](docs/AUDITORIA_INTERNA_FASE3_9.md).
A auditoria foi executada no Colab conforme resumo fornecido pelo grupo:
o sampler destinou 7,835% dos sorteios a reais TGIF e 1,740% a sintéticas
TGIF; na seleção TGIF, a 3.9 acertou 13/128 sintéticas e 114/128 reais.
Nas 1.420 imagens internas comuns, a acurácia ficou em 0,9211 para a 3.9 e
0,9176 para a 3.8. O próximo experimento isolado, se aprovado, terá o nome
**Fase 3.9.1**; futuras variantes seguem 3.9.2, 3.9.3 etc., preservando a
3.9 e seus resultados. O próximo passo é definir uma única alteração interna
para a 3.9.1 e medir seu efeito em FP, FN e acurácia total, preservando o
Mirage comparativo já observado.
Como a célula #10 mostra apenas erros extremos, a célula **#11** foi adicionada
ao mesmo notebook para contar todos os 59 FP e 182 FN por fonte/classe, grupo,
formato e resolução, usando somente o CSV interno já salvo. O grupo executou
a célula e informou a contagem completa: 115/128 sintéticas TGIF foram FN;
os FP vieram de Community Forensics (18), OpenFake (16), TGIF (14) e AIGI
(11). Isso soma 241 erros/1.676 imagens na seleção que escolheu o checkpoint;
não comprova acurácia externa. A célula não treina nem abre o Mirage.

**Atualização após o piloto TGIF:** a saída do Colab informou 341 IDs
originais/sintéticos compartilhados em `validation`, sem treino, calibração
ou avaliação externa. O piloto verificou os dois arquivos completos, oito
amostras PNG de cada e nenhum membro desconhecido. O TGIF2 teve apenas os
tamanhos dos arquivos de treino consultados. A 3.9 TGIF SD2-fr foi preparada
em código isolado. O preparo, smoke e treino interno foram concluídos no
notebook; a avaliação comparativa Mirage da 3.9 também foi concluída. Procedimento em
[`docs/README_fase3_9_execucao.md`](docs/README_fase3_9_execucao.md).

**Treino interno 3.9 concluído:** o Colab congelou 768 pares TGIF e confirmou
7.176 imagens de treino, 1.676 de seleção e 1.676 de calibração; o teste
OpenFake de 1.947 imagens foi apenas conferido por hash. O smoke passou com
GPU Tesla T4 e checkpoint inicial da 3.3. Foram executadas 12 épocas;
a melhor foi a 10, com AUC de seleção **0,8988236**. A calibração interna
registrou acurácia balanceada aparente **0,8557320**, temperatura **0,9187889**
e limiar **0,47**. O notebook confirmou `phase3_final.pt`, `best.pt`,
`internal_metrics.json` e `training_complete.json` no Drive. Nenhuma
inferência no teste OpenFake reservado foi registrada. As células #11 a #14
do notebook da 3.9 foram concluídas; não repetir preparo, treino ou avaliação.

**Comparação Mirage 3.9 concluída em 05/11/2026:** a saída salva no notebook
registra 2.000 imagens, acurácia **0,5875**, AUC **0,6328105**, recall
sintético **0,431**, especificidade real **0,744** e F1 **0,5109662**;
TN=744, FP=256, FN=569 e TP=431. O checkpoint, a temperatura 0,9187889
e o limiar 0,47 foram congelados antes da inferência. Na comparação pareada
com a 3.8, houve 1.063 acertos conjuntos, 135 acertos exclusivos da 3.8,
112 da 3.9 e 690 erros conjuntos. O resultado foi salvo no Drive em
`resultados/benchmark_externo_mirage_2000_v1/fase3_9_tgif_sd2fr_v1/`.
O Mirage já era conhecido: esta é uma comparação descritiva, não um teste
cego independente nem base para retreinar ou ajustar o limiar.

Atualizado em 05/11/2026 com as saídas revisadas dos notebooks e os resultados
informados pelo grupo. O Drive não foi montado nesta revisão; a existência
remota dos artefatos foi verificada pelas saídas do Colab, não por leitura
direta nesta máquina. O registro detalhado anterior está em
[`docs/historico/ESTADO_PROJETO_ate_3_6.md`](docs/historico/ESTADO_PROJETO_ate_3_6.md).

**Atualização 3.9:** o grupo executou a auditoria interna da 3.8 e informou
1.420 imagens de seleção, acurácia descritiva 0,9176056, AUC 0,9749367,
74 falsos positivos e 43 falsos negativos. O checkpoint manteve SHA-256
`de4a11cbb733ad9088860d5f254cbb2a900f181cc1ccb28b2fd894cf4a37706a`.
Os pilotos combinados Open Images + novos pares AIGI estão preparados em
`Notebooks/fase_3/pilotos_fase3_9_colab.ipynb` e
`experimentos/fase3_9_piloto_aigi.py` e foram executados pelo grupo. O
Open Images mostrou 1.739.728 candidatos elegíveis nos metadados, 76.961
autores declarados e dez JPEGs legíveis. O AIGI mostrou 129, 129 e 311 UIDs
novos pareáveis em FLUX.1_dev, PA_alpha e SD_2.1, mas somente 0, 0 e 2 fotos
reais utilizáveis nas cinco primeiras provas por grupo. A prova ampliada teve
0/20 fotos utilizáveis em FLUX.1_dev, 0/20 em PA_alpha e 5/20 em SD_2.1.
O grupo decidiu verificar TGIF/TGIF2 no lugar de ampliar AIGI. O piloto de
acesso e pareamento TGIF foi executado em
`Notebooks/fase_3/piloto_fase3_9_tgif_colab.ipynb`.
Ver `docs/README_fase3_9_execucao.md`. O treino interno já foi concluído;
a avaliação comparativa Mirage da 3.9 está concluída.

## Histórico da Fase 3.8 e contexto da 3.9

- **Branch:** `fase3-ajuste-fino`, acompanhando `origin/fase3-ajuste-fino` no
  início desta revisão. Em outro computador, confirme o commit com
  `git status -sb` e `git rev-parse HEAD` antes de abrir o Colab.
- **Objetivo:** detector de imagens reais e sintéticas com pelo menos 85% de
  acurácia em teste externo independente ainda não observado. Essa meta não
  foi demonstrada. Vídeos e aplicação web/API permanecem para etapa futura.
- **Fase 3.8 (histórica):** ajuste fino com pares AIGI. O preparo foi concluído:
  **480 pares/960 imagens**, 80 pares por gerador, com 640 imagens adicionais
  no treino, 160 na seleção e 160 na calibração. Os dados estão em
  `/content/drive/MyDrive/TCC_Final/dados_fase3_8/aigi_pares_256_v1/`.
- **Cópia local:** a célula `# 8` confirmou as 960 imagens AIGI em
  `/content/aigi_pares_256_v1/`, além das cópias locais 3.5 e 3.7.
- **Smoke test:** a célula `# 9` terminou com `Smoke OK`, GPU, arquitetura,
  checkpoint inicial da **Fase 3.3**, forward e gravação. Conferiu por hash
  6.152 imagens de treino, 1.420 de seleção, 1.420 de calibração e 1.947 do
  teste OpenFake reservado. O teste reservado não foi inferido.
- **Treino 3.8:** a célula `# 10` concluiu 12 épocas no Colab. A melhor foi a
  época 12, com AUC de seleção **0,9603094**. A calibração interna registrou
  acurácia balanceada aparente **0,9113623**, temperatura **0,5229031** e
  limiar **0,31**. As células `# 11` e `# 15` confirmam os artefatos e o
  resultado salvo. O teste OpenFake reservado não foi inferido.
- **Avaliação Mirage 3.8:** 1.198/2.000 acertos (**59,9%**), AUC **0,641268**,
  recall sintético **55,7%** e especificidade real **64,1%**; TN=641, FP=359,
  FN=443, TP=557. O checkpoint congelado tem SHA-256
  `de4a11cbb733ad9088860d5f254cbb2a900f181cc1ccb28b2fd894cf4a37706a`.
  Resultado comparativo no Drive, sem ajuste por imagens externas.

## Resultados anteriores comparáveis

| Modelo | Mirage 2000: acurácia | AUC | Estado |
| --- | ---: | ---: | --- |
| Fase 3.5 | 60,45% | 0,679972 | Treino e avaliação concluídos |
| Fase 3.6 | 57,20% | 0,6303885 | Treino e avaliação concluídos |
| Fase 3.7 | 62,30% | 0,6908315 | Treino e avaliação concluídos |
| Fase 3.8 | 59,90% | 0,641268 | Treino e avaliação concluídos |

As métricas acima vieram das saídas do Colab enviadas pelo grupo. O Mirage
contém 1.000 imagens reais e 1.000 sintéticas fixas, mas já foi observado.
Seu uso em novas fases é comparativo; não usar suas respostas para treinar,
calibrar ou escolher pesos, dados ou hiperparâmetros. A avaliação independente
final exigirá outro conjunto reservado antes de ser visto. O benchmark
RAISE/MNW e as 20 imagens externas antigas também são históricos.

## Como continuar o projeto

O experimento 3.8 está concluído; não repetir `# 7` (preparo), `# 9` (smoke),
`# 10` (treino) nem `# 14` (avaliação). Para consultar as saídas, abra
[`Notebooks/fase_3/rodar_fase3_8_colab.ipynb`](Notebooks/fase_3/rodar_fase3_8_colab.ipynb)
e leia as células `# 11` e `# 15` já executadas. Em novo runtime, os resultados
persistidos estão no Drive nos caminhos abaixo; `/content` é temporário.

Auditoria **interna** executada pelo grupo com
[`Notebooks/fase_3/auditar_interno_fase3_8_colab.ipynb`](Notebooks/fase_3/auditar_interno_fase3_8_colab.ipynb)
e resultado resumido informado nesta conversa. Ela inspeciona dados de treino/seleção e
erros da seleção por fonte, gerador, resolução e formato; não abre imagens de
calibração/teste nem predições Mirage. As quatro alternativas originais para 3.9 estão em
[`docs/AUDITORIA_E_OPCOES_FASE3_9.md`](docs/AUDITORIA_E_OPCOES_FASE3_9.md).
A hipótese de novos pares AIGI foi interrompida após a baixa disponibilidade
de fotos reais. O piloto TGIF, o preparo versionado, o smoke, o treino, a
seleção, calibração e avaliação Mirage da 3.9 já terminaram. Não repetir
essas etapas. A próxima atividade é auditar **somente dados e erros internos**
da 3.9, especialmente a contribuição dos pares TGIF e atalhos de resolução,
formato e conteúdo. O Mirage já foi observado; suas respostas não podem
orientar outro ajuste.

Os checkpoints da 3.8 ficam em
`/content/drive/MyDrive/TCC_Final/modelos/fase3_8_aigi_pares256_v1/`;
resultados comparativos em
`/content/drive/MyDrive/TCC_Final/resultados/benchmark_externo_mirage_2000_v1/fase3_8_aigi_pares256_v1/`.
O checkpoint final da 3.9 está em
`/content/drive/MyDrive/TCC_Final/modelos/fase3_9_tgif_sd2fr_v1/phase3_final.pt`.
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
- Comparada à 3.7 nas mesmas imagens Mirage, a 3.8 acertou **48 imagens a
  menos**: 180 acertos exclusivos da 3.7 e 132 exclusivos da 3.8. O recall
  sintético subiu de 44,0% para 55,7%, enquanto a especificidade real caiu de
  80,6% para 64,1%. Esse resultado não autoriza ajustar o limiar no Mirage.
- O log do treino imprime `Partições 3.7` por uma mensagem herdada. O código
  da 3.8 acrescenta os pares AIGI, e a auditoria do treino mostra 320 reais
  e 320 sintéticas AIGI. Preservar o script usado nesta execução e seu hash;
  corrigir a mensagem em uma versão futura, sem reescrever a proveniência.
- Escolher a próxima hipótese **apenas com dados internos**: diversidade de
  novos datasets, qualidade dos rótulos e controle de atalhos de tamanho e
  formato têm prioridade. O teste Mirage relata generalização observada e
  não determina ajustes da próxima versão.

Para entender a arquitetura e as métricas, leia
[`docs/GUIA_METODOLOGIA_TCC.md`](docs/GUIA_METODOLOGIA_TCC.md). Para detalhes
de preparo, retomada e avaliação da 3.8, leia
[`docs/README_fase3_8_execucao.md`](docs/README_fase3_8_execucao.md).
