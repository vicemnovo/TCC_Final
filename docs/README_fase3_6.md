# Fase 3.6 — hipótese isolada sobre pistas de escala

## Evidência e hipótese

Na seleção interna da 3.5, as 312 imagens sintéticas do Community
Forensics–Small têm lado curto processado entre 512 e 1.023 px; 207 das
225 reais desse dataset têm lado curto abaixo de 512 px. Há portanto uma
associação entre classe e resolução **nos arquivos processados**, mas a
auditoria não prova que o modelo a usa como atalho. Os 12 erros de fotos
reais do OpenFake com lado curto de pelo menos 512 px e os falsos negativos
fotográficos de alta confiança motivam um teste de robustez limitado.

A hipótese da 3.6 é que expor ambas as classes com mais frequência a
redimensionamento moderado durante o treino reduza a dependência de pistas
de escala e melhore a seleção interna sem prejudicar os sinais visuais.
O efeito pode ser nulo ou negativo: downsampling também pode apagar
artefatos úteis. Não declarar melhoria antes de medir.

## Única mudança experimental

`experimentos/fase3_6_escala.py` copia o fluxo de ajuste fino da 3.5 e
altera apenas a probabilidade de `RandomDownUp` no treino de **0,15 para
0,50**; o intervalo de escala continua **0,5–0,9**. A transformação vale
para ambas as classes. A arquitetura ViT-Tiny + FFT + EfficientNet-B0,
o checkpoint inicial da 3.3, os dados congelados, a divisão 3.5, a
amostragem, as demais transformações, o otimizador, as taxas, o cronograma,
12 épocas, seleção por AUC e calibração interna são os mesmos da 3.5.
A transformação de inferência de cinco recortes também não muda.

O script compara o registro de partições da 3.6 com
`modelos/fase3_5_cfsmall_v1/split_manifest.json` e recusa executar se
diferirem. O teste OpenFake histórico é apenas conferido por hash, sem
inferência. RAISE/MNW e as 20 imagens antigas não participam do treino,
seleção, calibração nem escolha de parâmetros.

## Execução no Colab

Abra `Notebooks/fase_3/rodar_fase3_6_colab.ipynb` em GPU. O notebook não
treina ao abrir e reutiliza a credencial GitHub e os dados já congelados.

1. Execute **1–5**: GPU/dependências, Drive, senha, snapshot do GitHub e
   comandos. Confirme que o snapshot contém `fase3_6_escala.py`.
2. Execute **6** para copiar e verificar as imagens internas no `/content`.
3. Execute **7** para o smoke: hashes, partições idênticas, checkpoint e
   forward. Só prossiga se aparecer `Smoke OK`.
4. Execute **8 manualmente** para iniciar ou retomar o treino. Ele salva em
   `modelos/fase3_6_escala_p05_v1/`, separado da 3.5. Não execute esta
   célula durante a preparação do projeto.
5. Execute **9** para ler os artefatos e envie as métricas internas.

O modo `auto` não prolonga um experimento concluído. Se o Colab desconectar,
refaça 1–6 e execute 8; os checkpoints `last.pt` e `best.pt` preservam a
retomada após uma época completa. O smoke usa diretório separado
`modelos/fase3_6_escala_p05_v1_smoke/`.

## Comparação e decisão

O usuário informou treino concluído na GPU Tesla T4, com as mesmas
partições da 3.5 (3.594 treino, 719 seleção, 719 calibração; 1.947 do teste
histórico apenas conferidas por hash). O checkpoint inicial da Fase 3.3
foi carregado e os oito artefatos esperados foram informados como presentes
em `modelos/fase3_6_escala_p05_v1/`. O Drive não foi inspecionado diretamente
nesta sessão local.

| Indicador interno | Fase 3.5 | Fase 3.6 |
| --- | ---: | ---: |
| Melhor época | 10 | 12 |
| AUC de seleção, recorte central | 0,9487973 | 0,9472355 |
| Acurácia balanceada aparente de calibração | 0,9212768 | 0,9075869 |
| Temperatura | 0,4409471 | 0,4662258 |
| Limiar | 0,445 | 0,495 |

A diferença da AUC é -0,00156, pequena para sustentar uma conclusão
causal com uma execução por variante. A acurácia balanceada da calibração
foi obtida na mesma partição usada para ajustar o limiar e é aparente.
Esses números não medem generalização externa. Manter a 3.5 como candidata
interna preferida, preservar a 3.6 e não repetir a célula 8 para buscar
mais épocas: `training_complete.json` marca a execução como concluída.
Por decisão do projeto, a avaliação externa comparativa desta fase usará o
**mesmo Mirage-Test 2000 v1** já congelado e avaliado na 3.5. Isso permite
comparação pareada nas mesmas 1.000 fotos reais e 1.000 sintéticas, com
temperatura e limiar próprios de cada checkpoint. Como o conjunto já foi
observado, o resultado da 3.6 **não é um novo teste cego independente** e
não deve orientar treino, escolha de pesos ou hiperparâmetros da 3.7.

## Avaliação externa comparativa da 3.6: concluída conforme saída do Colab

O usuário informou **1.144/2.000 acertos (57,2%)**, AUC **0,6303885**,
recall sintético **28,8%** e especificidade real **85,6%**. Na mesma seleção
congelada, a 3.5 obteve **1.209/2.000 (60,45%)** e AUC **0,679972**.
A comparação pareada registrou 1.123 imagens acertadas por ambas,
86 somente pela 3.5, 21 somente pela 3.6 e 770 erradas por ambas.
O Drive não foi montado nesta revisão local; estes números vêm da saída
enviada pelo usuário. Não usar esses casos para ajustar a Fase 3.7.

### Procedimento já executado, preservado para reprodutibilidade

Abra `Notebooks/fase_3/avaliar_fase3_6_mirage_2000_colab.ipynb` no Colab
com GPU e execute, em ordem:

1. Células **1–5**: dependências, montagem do Drive, desbloqueio da
   credencial GitHub, código da branch e função de progresso. O snapshot
   deve incluir `experimentos/avaliar_fase3_6_mirage.py`.
2. Célula **6**: copia as 2.000 imagens do Drive para `/content` e verifica
   seus hashes; não refaz o download das fontes nem o plano congelado.
3. Célula **7**: congela, antes da inferência, o hash do manifesto, o
   checkpoint 3.6, temperatura **0,4662258029**, limiar **0,495**, código
   de avaliação e resultado anterior da 3.5. Não altera esses arquivos.
4. Célula **8**: inicia manualmente a inferência externa. Usa cinco recortes
   de 224 px e salva em `resultados/benchmark_externo_mirage_2000_v1/
   fase3_6_escala_p05_v1/` os arquivos `metrics.json`, `predictions.csv`
   e `per_group.json`. A comparação pareada registra casos corrigidos e
   piorados em relação à 3.5. Um resultado existente não é sobrescrito.
5. Célula **9**: lê as métricas das duas fases, sem nova inferência.

Não execute piloto, preparação ou treino novamente. Se a sessão cair antes
da célula 8, repita as células 1–6; a célula 7 reconhece um bloqueio
idêntico já salvo. Se cair durante a gravação dos resultados, uma pasta
`.incompleto.*` pode permanecer no Drive; o diretório final só aparece
depois que os três arquivos estiverem escritos. O resultado informado pelo
usuário já foi obtido; não repetir a célula 8 para selecionar versões.
