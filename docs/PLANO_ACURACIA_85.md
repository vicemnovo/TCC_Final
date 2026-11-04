# Plano atual — meta de 85% de acurácia externa

**Estado verificado em 03/11/2026:** a 3.7 concluiu treino e avaliação
comparativa; obteve 62,3% no Mirage 2000 conforme saída do Colab. A 3.8
concluiu preparo, cópia e smoke test com 480 pares AIGI. Seu treino e sua
avaliação ainda não aparecem no notebook versionado. A meta de 85% continua
aberta. Consulte [`ESTADO_PROJETO.md`](../ESTADO_PROJETO.md) para a retomada.

## Próxima entrega: concluir a Fase 3.8

1. **Treino interno:** executar manualmente a célula `# 10` de
   `Notebooks/fase_3/rodar_fase3_8_colab.ipynb`. O treino parte dos pesos
   da 3.3 e usa OpenFake, Community Forensics e os novos pares AIGI. Salva
   `last.pt` para retomada e `best.pt` para a melhor época. Conferir perdas,
   AUC de seleção e o arquivo `training_complete.json`.
2. **Seleção e calibração:** a época é escolhida pela partição interna de
   seleção. Temperatura e limiar são ajustados na partição interna de
   calibração. Registrar as métricas com a célula `# 11`. Esses números não
   demonstram desempenho em imagens externas.
3. **Avaliação comparativa:** executar as células `# 12` a `# 15`. Elas
   copiam o Mirage para `/content`, congelam checkpoint, temperatura e
   limiar e calculam acurácia, AUC, recall sintético, especificidade real,
   F1 e matriz de confusão nas mesmas 2.000 imagens usadas nas fases 3.5–3.7.
   Salvar o resultado em diretório próprio da 3.8. O Mirage já foi observado:
   relatar o resultado sem usá-lo para alterar a 3.8 ou planejar parâmetros
   da próxima versão.
4. **Registro:** atualizar `ESTADO_PROJETO.md` com melhor época, métricas
   internas, checkpoint, hashes e resultado externo **somente após receber
   as saídas reais**. Preservar os dados e resultados das fases anteriores.

## Próxima hipótese após a 3.8

Priorizar **novos dados internos úteis para generalização**, conforme a visão
do grupo. A auditoria do smoke da 3.8 encontrou uma associação importante:
1.936 reais Community Forensics tinham lado curto de 256–511 px, enquanto
1.744 sintéticas tinham 512 px ou mais. Os 320 pares AIGI usados no treino
estão equilibrados em formato e faixa de tamanho, mas não eliminam essa
associação do conjunto combinado. Antes de qualquer Fase 3.9:

1. Analisar erros e desempenho **na seleção interna**, por fonte,
   gerador, formato, resolução e tipo de cena. A análise indica hipóteses;
   não prova que a rede tenha usado um atalho.
2. Escolher **uma** mudança de dados justificada: mais variedade de cenas,
   geradores ou fotografias reais com rótulos confiáveis, ou controle melhor
   da associação classe × tamanho/formato. Fixar versão, divisões por grupo
   ou UID, quantidade e critérios de exclusão antes de treinar.
3. Fazer piloto de licença, download, espaço e legibilidade; baixar os
   arquivos escolhidos integralmente para `/content`, sem streaming durante
   treino ou validação. Guardar seleção e manifesto de hashes no Drive.
4. Criar experimento e diretório novos; fazer smoke, ajuste fino e
   calibração interna. Preservar os checkpoints 3.3–3.8. Avaliar no Mirage
   só depois do congelamento, como comparação descritiva.

Mudanças de arquitetura, mais recortes ou uma pista forense específica de
ruído, cor, pixel ou frequência continuam possíveis, mas exigem evidência
interna de utilidade e custo de GPU justificável. O ramo FFT atual oferece
informação de frequência; não comprova sozinho a detecção de um artefato
particular ou de incoerência física.

## Como interpretar as etapas

| Etapa | O que produz | O que ainda não comprova |
| --- | --- | --- |
| Piloto | Viabilidade de fonte, formato, licença e download | Acurácia do modelo |
| Preparo/congelamento | Imagens escolhidas, divisões e hashes no Drive | Qualidade das previsões |
| Smoke test | Dados acessíveis, checkpoint compatível e forward finito | Treino ou generalização |
| Treino | Pesos, histórico e checkpoints recuperáveis | Desempenho externo |
| Seleção interna | Época escolhida sem olhar o conjunto externo | Teste independente |
| Calibração interna | Temperatura e limiar congeláveis | Probabilidade confiável em toda fonte nova |
| Avaliação Mirage | Comparação entre versões nas mesmas imagens | Teste cego independente após feedback prévio |
| Retomada | Continuação de uma execução incompleta via `last.pt` | Novo experimento ou mais épocas após conclusão |

## Critério de sucesso

Comparar internamente AUC, acurácia balanceada, recall sintético,
especificidade real, precisão, F1 e desempenho por grupo; registrar também
tempo e custo de armazenamento. Acurácia Mirage acompanha cada versão,
sem selecionar dados, pesos, limiar ou hiperparâmetros. Para afirmar
**85% em teste externo independente** de uma versão criada após observar
Mirage, reservar e congelar outro conjunto ainda não usado no desenvolvimento.
Não prometer essa taxa antes da avaliação.
