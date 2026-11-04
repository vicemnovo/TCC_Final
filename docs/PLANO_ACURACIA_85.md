# Plano atual — meta de 85% de acurácia externa

**Estado em 04/11/2026:** a 3.7 obteve 62,3% e a 3.8 obteve 59,9% nas mesmas
2.000 imagens Mirage, conforme saídas do Colab. A 3.8 concluiu treino interno
com 480 pares AIGI adicionais e avaliação comparativa; a meta de 85% em
conjunto independente continua aberta. Consulte
[`ESTADO_PROJETO.md`](../ESTADO_PROJETO.md) para a situação atual.

## Fase 3.8 concluída: registro das etapas

1. **Treino e seleção internos:** o ajuste fino partiu da 3.3 e usou
   OpenFake, Community Forensics e pares AIGI. A melhor época foi a 12,
   com AUC de seleção 0,9603094. Os arquivos `last.pt`, `best.pt` e
   `training_complete.json` foram informados como presentes.
2. **Calibração interna:** temperatura 0,5229031 e limiar 0,31 foram
   definidos sem imagens externas. A acurácia balanceada aparente 0,9113623
   não estima desempenho em fontes novas.
3. **Avaliação comparativa Mirage:** checkpoint e calibração foram congelados
   antes da inferência. Acurácia 59,9%, AUC 0,641268, recall sintético 55,7%
   e especificidade real 64,1%. A 3.7 teve 62,3% e AUC 0,6908315 nas mesmas
   imagens. Preservar os dois resultados; não reajustar a 3.8 por esse teste.
4. **Próxima entrega:** análise de erros e distribuição **internas** antes
   de fixar uma única hipótese para a 3.9. O script e notebook estão
   preparados, mas a auditoria ainda não foi executada no Colab. Ver
   [`AUDITORIA_E_OPCOES_FASE3_9.md`](AUDITORIA_E_OPCOES_FASE3_9.md).
   Não repetir o treino concluído.

## Próxima hipótese após a 3.8

Priorizar **novos dados internos úteis para generalização**, conforme a visão
do grupo. A auditoria de cabeçalhos da 3.8 encontrou uma associação importante:
1.936 reais Community Forensics tinham lado curto de 256–511 px, enquanto
1.744 sintéticas tinham 512 px ou mais. Os 320 pares AIGI usados no treino
estão equilibrados em formato e faixa de tamanho, mas não eliminam essa
associação do conjunto combinado. O resultado Mirage não prova que o modelo
usou esse atalho. Antes de qualquer Fase 3.9:

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
