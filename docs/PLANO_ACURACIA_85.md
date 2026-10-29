# Plano de execução — meta de 85% de acurácia externa

## Estado comprovado

- A Fase 3.5 concluiu o ajuste fino interno e preservou o checkpoint em
  `modelos/fase3_5_cfsmall_v1/phase3_final.pt` no Google Drive.
- Na seleção interna já usada para escolher o modelo, cinco recortes deram
  90,68% de acurácia, com 26 falsos positivos e 41 falsos negativos.
- A Fase 3.5 ainda não foi medida em um conjunto externo independente.
  A melhor acurácia histórica no benchmark RAISE/MNW visto é 62,1% da
  Fase 3.4 v2. Esses números vêm de conjuntos diferentes e não são
  uma comparação direta entre os dois modelos.
- A Fase 3.6 também concluiu o treino interno, nas mesmas partições da 3.5.
  AUC de seleção: 0,9472355 contra 0,9487973 da 3.5; calibração aparente:
  0,9075869 contra 0,9212768. A variante de escala não demonstrou ganho
  interno. Preservar os dois checkpoints e usar a 3.5 como candidata atual
  para uma avaliação nova e independente.
- A meta de 85% é experimental e não pode ser garantida por número de
  épocas, temperatura, limiar ou volume de imagens isoladamente.

## Sequência prioritária

1. **Congelar a candidata 3.5 e definir um novo teste externo.** A auditoria
   interna e a variante 3.6 já foram executadas. Registrar pesos, temperatura
   0,4409471, limiar 0,445 e protocolo de cinco recortes da 3.5. Definir
   origem, rótulos, quantidade e critérios de exclusão do novo teste antes
   de produzir predições. RAISE/MNW e as 20 imagens históricas já são vistas.
2. **Medir a versão congelada em novo teste externo independente.**
   Conferir integridade e ausência de sobreposição com as divisões internas,
   fazer uma única inferência com a configuração congelada e salvar os
   resultados em diretório novo. Reportar acurácia com intervalo de
   confiança, recall sintético, especificidade real, precisão, F1, AUC e
   desempenho por grupo. Se o resultado orientar trabalhos posteriores,
   reservar outro conjunto independente para a avaliação final deles.
3. **Se necessário, desenvolver outra variante com dados internos.**
   Priorizar fontes reais de origem identificável e sintéticas fotográficas,
   porque a auditoria interna mostrou erros nessas categorias; confirmar
   proveniência e separação por fonte antes de treinar. Definir uma única
   mudança e critérios internos antes do treino. Baixar os arquivos
   escolhidos para `/content`, persistir manifesto no Drive e verificar
   duplicatas e vazamento. Reservar outro teste externo final para a nova
   versão; não usar as respostas do teste do passo 2 para ajustá-la.

## Critério para não perder tempo

Não estender as 12 épocas concluídas nem recalibrar o limiar da 3.5 com
conjuntos externos. A AUC de seleção melhorou até a época 10; mais épocas,
sem hipótese sobre os erros, não atacam a lacuna de generalização externa.
O desenvolvimento da aplicação web/API permanece adiado enquanto esta
etapa prioriza a acurácia do detector.
