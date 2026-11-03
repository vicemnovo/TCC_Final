# Plano de execução — meta de 85% de acurácia externa

## Estado comprovado

- A Fase 3.5 concluiu o ajuste fino interno e preservou o checkpoint em
  `modelos/fase3_5_cfsmall_v1/phase3_final.pt` no Google Drive.
- Na seleção interna já usada para escolher o modelo, cinco recortes deram
  90,68% de acurácia, com 26 falsos positivos e 41 falsos negativos.
- A Fase 3.5 foi medida no Mirage-Test externo congelado: **60,45%** de
  acurácia, AUC **0,679972** e recall sintético **35,1%** em 2.000 imagens.
  O RAISE/MNW visto registrou 62,1% para a 3.4 v2, em outro conjunto;
  esses números não formam uma comparação direta entre modelos.
- A Fase 3.6 também concluiu o treino interno, nas mesmas partições da 3.5.
  AUC de seleção: 0,9472355 contra 0,9487973 da 3.5; calibração aparente:
  0,9075869 contra 0,9212768. A variante de escala não demonstrou ganho
  interno. Preservar os dois checkpoints e usar a 3.5 como candidata atual
  para o diagnóstico já concluído; não usar o Mirage-Test para escolher
  retrospectivamente outra candidata.
- A meta de 85% é experimental e não pode ser garantida por número de
  épocas, temperatura, limiar ou volume de imagens isoladamente.

## Sequência prioritária

1. **Encerrar o Mirage-Test como avaliação da 3.5.** O checkpoint e os
   resultados estão congelados. Usar suas métricas apenas para descrever a
   generalização já observada; não escolher nova configuração com suas
   predições por imagem ou grupos.
2. **Desenvolver uma única variante com dados internos.** A auditoria
   interna anterior já mostrou origem real `n/a`, associação entre classe
   e resolução e sintéticas fotográficas difíceis. Priorizar fonte real
   identificável e separação por origem; definir a seleção e os critérios
   internos antes de treinar. Baixar os arquivos escolhidos para `/content`,
   persistir manifesto no Drive e verificar duplicatas e vazamento. O primeiro
   gate preparado é o [piloto Open Images da possível Fase 3.7](README_fase3_7_piloto.md):
   ele ainda não foi executado e não autoriza seleção ou treino automático.
3. **Reservar outro teste externo para a nova variante.** Definir origem,
   rótulos, quantidade, hashes e exclusões antes da inferência; congelar
   modelo, temperatura, limiar e protocolo. O Mirage-Test, RAISE/MNW e as
   20 imagens históricas já são vistos e não servem como avaliação final
   independente do próximo modelo.

## Critério para não perder tempo

Não estender as 12 épocas concluídas nem recalibrar o limiar da 3.5 com
conjuntos externos. A AUC de seleção melhorou até a época 10; mais épocas,
sem hipótese sobre os erros, não atacam a lacuna de generalização externa.
O desenvolvimento da aplicação web/API permanece adiado enquanto esta
etapa prioriza a acurácia do detector.
