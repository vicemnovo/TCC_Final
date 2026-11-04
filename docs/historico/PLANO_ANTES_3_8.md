# Plano anterior à Fase 3.8 — registro histórico

Este roteiro foi escrito antes do experimento AIGI da 3.8. As seções que
descrevem 3.7 e 3.8 como futuras não são instruções de execução atuais.
Para continuar, consulte `../../ESTADO_PROJETO.md` e
`../PLANO_ACURACIA_85.md`.

## Texto preservado

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
- A avaliação comparativa Mirage da 3.6 foi informada como concluída:
  acurácia **57,2%**, AUC **0,6303885**, recall sintético **28,8%** e
  especificidade real **85,6%**. A 3.5 acertou 86 imagens que a 3.6 errou;
  a 3.6 acertou 21 que a 3.5 errou. Esses resultados documentam a variante,
  mas não devem escolher os dados ou parâmetros da próxima versão.
- A meta de 85% é experimental e não pode ser garantida por número de
  épocas, temperatura, limiar ou volume de imagens isoladamente.

## Roteiro da Fase 3.7 em diante

**Princípio:** a entrada de produção é a própria imagem, não sua URL, autor,
metadados ou histórico. “Copigmentação” é tratada provisoriamente como
**consistência de cor/pigmentação**. Ruído, padrões locais de pixel, coerência
global/física e frequências são hipóteses de pistas úteis, não diagnósticos
que a arquitetura atual já comprove. A Fase 2/3 já combina ViT-Tiny, CNN
EfficientNet-B0 e um ramo FFT que usa a magnitude do espectro; a inferência
Mirage da 3.5/3.6 usa cinco recortes de 224 px. O treino já inclui JPEG,
desfoque e `RandomDownUp`; a 3.6 mostrou que intensificar apenas este último
não trouxe ganho interno nem no Mirage comparativo.

### 3.7 — dados e rótulos, sem mudar a arquitetura

1. **Piloto de fonte:** verificar licença, número de arquivos, espaço,
   proveniência dos rótulos e variedade de cenas reais, geradores sintéticos
   e imagens editadas. Baixar arquivos escolhidos integralmente para
   `/content`; o [piloto Open Images](../README_fase3_7_piloto.md) é opcional.
   Piloto mede viabilidade, não treina nem prova acurácia.
2. **Preparo congelado:** fixar a seleção e um manifesto com versão, hashes,
   classe, grupo, formato, resolução e tipo de conteúdo quando conhecido.
   Guardar cópia no Drive e verificar duplicatas exatas e candidatos visuais
   entre treino, seleção, calibração e testes. Não usar Mirage, RAISE/MNW ou
   as 20 imagens antigas nessa seleção. Separar grupos/autores/geradores
   quando houver informação; não inventar origem ausente. Evitar que formato,
   compressão ou resolução identifiquem a classe por si. Manter rótulos
   verificáveis sem exigir metadados no momento de inferir.
3. **Smoke test:** copiar o conjunto persistente para `/content`, conferir
   hashes, GPU, partições, arquitetura, checkpoint inicial e uma passagem
   forward. Isso detecta falhas de execução; não estima desempenho.
4. **Ajuste fino:** manter ViT + FFT + CNN e o protocolo de treino anterior
   como base, alterando só a composição dos dados. Comparar checkpoints
   existentes por evidência interna antes de escolher o inicial. Salvar
   `last.pt`, `best.pt`, histórico e configuração em diretório 3.7 novo.
   Selecionar melhor época e calibrar temperatura/limiar apenas em partições
   internas apropriadas; se um checkpoint anterior foi selecionado com uma
   partição antiga, não tratá-la como validação independente nova.
5. **Avaliação:** congelar pesos, temperatura, limiar e protocolo; avaliar
   uma vez o Mirage 2000 para comparação descritiva, em pasta própria.
   Registrar acurácia, AUC, recall sintético, especificidade real, F1,
   acurácia balanceada e matriz de confusão. Não usar essas predições para
   ajustar a 3.8.

### 3.8 — testar contexto global e consistência entre recortes

Se a seleção **interna** mostrar erros ligados ao contexto da cena, testar
uma variante isolada que acrescente uma vista da imagem inteira aos cinco
recortes, com custo de inferência medido no Colab. Comparar internamente a
decisão por recorte e a dispersão das probabilidades; calibrar a combinação
somente com dados internos. Uma vista global pode oferecer contexto sobre
objetos, sombras e relações espaciais, mas não demonstra raciocínio físico.
Preservar o modelo 3.7 e fazer smoke, treino somente se necessário,
calibração e avaliação Mirage em diretórios 3.8 separados. Se não houver
ganho interno consistente, não gastar GPU nessa variante.

### 3.9 — uma pista forense explícita, somente se justificada

Usar a auditoria interna da 3.7/3.8 para escolher **uma** hipótese: resíduos
locais/ruído, correlações de cor ou representação de frequência. Antes de
adicionar ramo ou alterar a FFT, testar se a pista discrimina classes em
vários geradores, fontes, formatos e resoluções, inclusive após compressão.
Um resultado restrito a um formato ou fonte é um atalho, não evidência de
generalização. Só implementar a mudança com ganho interno reproduzível e
custo de GPU aceitável. Coerência física explícita exigiria dados internos
rotulados para essas violações e uma avaliação específica; não atribuir essa
capacidade automaticamente ao ViT. Preservar versões anteriores e repetir
o protocolo externo comparativo congelado.

### Decisão final

Usar critérios internos predefinidos para avançar entre versões: AUC e
acurácia balanceada de seleção, recall sintético, especificidade real,
calibração de probabilidades, desempenho por grupo e estabilidade a JPEG,
WebP, desfoque, cor e resolução. Acurácia Mirage é **relato externo** de cada
versão, não critério de escolha de dados, hiperparâmetros ou checkpoint.
Para declarar a meta de 85% em teste cego independente de uma versão criada
após observar Mirage, reservar outro conjunto externo ainda não visto.

## Critério para não perder tempo

Não estender as 12 épocas concluídas nem recalibrar o limiar da 3.5 com
conjuntos externos. A AUC de seleção melhorou até a época 10; mais épocas,
sem hipótese sobre os erros, não atacam a lacuna de generalização externa.
O desenvolvimento da aplicação web/API permanece adiado enquanto esta
etapa prioriza a acurácia do detector.
