# Auditoria interna da Fase 3.9

## Pergunta

A Fase 3.9 acrescentou 768 pares TGIF original/SD2-fr, mas a acurácia
comparativa Mirage informada pelo Colab ficou em 58,75%, contra 59,9% da
3.8. Esta diferença observada não identifica, por si, a causa da queda.
O Mirage já foi visto e não pode orientar seleção de pesos, dados, limiar
ou hiperparâmetros. Esta auditoria usa apenas treino e seleção internos.

## Explicações possíveis, ainda não confirmadas

1. **Mudança nas decisões após a calibração.** O limiar interno passou de
   0,31 na 3.8 para 0,47 na 3.9 e a temperatura também mudou. No Mirage,
   o recall sintético caiu de 0,557 para 0,431, enquanto a especificidade
   real subiu de 0,641 para 0,744. Esse padrão é compatível com decisões
   mais conservadoras para a classe sintética, mas as distribuições de
   pontuação também mudaram. A AUC diminuiu de 0,641268 para 0,6328105,
   então o limiar sozinho não explica toda a diferença observada.
2. **Peso e domínio dos pares TGIF.** A 3.9 acrescentou 1.024 imagens TGIF
   ao treino. O sampler usa massa por grupo proporcional à raiz do tamanho
   do grupo, de modo que a participação efetiva do TGIF pode diferir de sua
   fração bruta no dataset. A fonte COCO/SD2-fr pode ensinar pistas úteis
   para si e menos transferíveis a outros tipos de imagem. A auditoria
   calculou a massa esperada do sampler e os acertos no subconjunto TGIF.
3. **Atalhos nos dados históricos.** No smoke da 3.9, a maior parte das
   fotos reais Community Forensics ficou na faixa de lado curto 256–511 px,
   enquanto a maioria das sintéticas dessa fonte ficou em 512 px ou mais.
   Os pares TGIF foram normalizados em ambas as classes, mas não removem
   essa associação anterior. O modelo pode continuar explorando resolução,
   formato ou conteúdo específico da fonte; a auditoria mede associações,
   sem alegar que uma delas causou o erro externo.

A diferença de acurácia Mirage entre 3.8 e 3.9 é de **1,15 ponto percentual**.
São 135 imagens acertadas apenas pela 3.8 e 112 apenas pela 3.9. Isso não
autoriza corrigir o modelo a partir dessas imagens já observadas.

## Frequência de FP e FN na seleção interna

A contagem completa informada pelo grupo soma **59 FP e 182 FN**. Para
avaliar a meta de acurácia total, uma futura variante precisa considerar
os dois lados; apenas aumentar o recall sintético pode trocar FN por FP.
São 241 erros/1.676 imagens e, portanto, 1.435 acertos (**85,62%**) nessa
seleção interna. Esse número não atende a meta externa: a seleção participou
da escolha do checkpoint e, no Mirage comparativo já observado, a 3.9 obteve
58,75%. No conjunto interno, o recall sintético foi 736/918 (**80,17%**) e a
especificidade real, 699/758 (**92,22%**). Melhorar um lado com grande perda
no outro pode reduzir a acurácia total.

| Classe e fonte | Erros / imagens | Taxa de erro | Parcela dos erros da classe |
| --- | ---: | ---: | ---: |
| Sintéticas TGIF — FN | 115 / 128 | 89,8% | 63,2% dos FN |
| Sintéticas Community Forensics — FN | 56 / 628 | 8,9% | 30,8% dos FN |
| Sintéticas OpenFake — FN | 10 / 82 | 12,2% | 5,5% dos FN |
| Sintéticas AIGI — FN | 1 / 80 | 1,2% | 0,5% dos FN |
| Reais Community Forensics — FP | 18 / 450 | 4,0% | 30,5% dos FP |
| Reais OpenFake — FP | 16 / 100 | 16,0% | 27,1% dos FP |
| Reais TGIF — FP | 14 / 128 | 10,9% | 23,7% dos FP |
| Reais AIGI — FP | 11 / 80 | 13,8% | 18,6% dos FP |

Os cinco FP AIGI na galeria da célula #10 são **erros mais confiantes**, não
os FP mais numerosos. Dos 115 FN TGIF, 84 tiveram `p_sintetica <= 0,20`;
isso mostra confiança alta em uma classificação errada dentro dessa fonte,
sem provar a causa. Na seleção comum às fases 3.8/3.9, a 3.9 ganhou cinco
acertos líquidos, mas trocou 29 FP a menos por 24 FN a mais. O diagnóstico
por fonte é mais útil que a acurácia interna isolada.

A primeira variante controlada, **Fase 3.9.1**, pode testar apenas o sampler
TGIF, preservando dados, divisões, arquitetura, checkpoint inicial e demais
parâmetros. Sua avaliação interna deverá mostrar FP, FN, AUC, acurácia total,
acurácia balanceada, recall sintético e especificidade real nas mesmas imagens.
Se ganhar recall TGIF à custa de muitos FP reais ou perdas nas outras fontes,
não será considerada melhoria. O Mirage já visto não pode definir esse ajuste.

## O que o notebook faz

Abra `Notebooks/fase_3/fase3_9/auditar_interno_fase3_9_colab.ipynb` em um runtime
Colab com GPU. Execute as células de código **#1 a #5** em ordem para montar
Drive, desbloquear a credencial GitHub e sincronizar a branch publicada.
Execute **#6** para copiar os dados internos congelados ao `/content` com
verificação de hashes. Execute **#7** para a auditoria; **#8** lê as tabelas,
**#9** mostra dois gráficos e **#10** permite revisão visual opcional dos
erros internos mais confiantes. A **#11** lê todas as 1.676 predições internas
já salvas e imprime frequências por fonte/classe, grupo, formato e resolução,
com contagem, denominador e taxa. Para essa célula, em uma nova sessão basta
montar o Drive pela #2: ela não precisa de GPU, código do GitHub ou nova
inferência. Nenhuma célula inicia treino ou recalibração.

O script principal `Notebooks/fase_3/fase3_9/auditoria_interna_fase3_9.py`:

- Confere hashes de checkpoints, scripts, arquitetura e manifestos de treino.
- Verifica que as divisões da 3.8 permanecem na 3.9 e que somente os 768
  pares TGIF foram acrescentados; cada ID real/sintético fica na mesma divisão.
- Lê os cabeçalhos de treino e seleção para comparar classe, fonte, formato,
  resolução, proporção da imagem e categoria TGIF. Calcula a massa esperada
  de amostragem por fonte e classe, pois a política usa raiz do tamanho do grupo.
- Infere com **ambos** os checkpoints nas mesmas 1.420 imagens da seleção
  anterior e nas 256 imagens TGIF da nova seleção. Usa cinco recortes e os
  valores de temperatura/limiar já congelados de cada modelo. Registra AUC,
  acurácia, erros por segmento e acertos pareados.

Os arquivos novos são gravados somente em
`/content/drive/MyDrive/TCC_Final/resultados/auditoria_interna_fase3_9_v1/`:
`summary.json`, `metadata_train_selection.csv`, `distribution.csv`,
`sampler_mass_train.csv`, `predictions_selection_paired.csv`,
`errors_selection_3_9.csv` e `segments_selection_3_9.csv`. A execução
recusa sobrescrever esse diretório; se o relatório já existir, apenas o lê.
O script não abre imagens da calibração, do teste OpenFake reservado ou do
Mirage. A seleção interna escolheu checkpoints; seus indicadores são
**descritivos** e não constituem um novo teste independente.

## Como ler o resultado

Compare primeiro `common_3_8` nos dois modelos: as imagens são idênticas.
Uma diferença de AUC indica mudança na ordenação das pontuações; uma mudança
de acurácia com AUC semelhante sugere examinar o efeito das decisões nos
limiares próprios, sem recalibrá-los com o Mirage. Leia `tgif_new` como
diagnóstico dos pares acrescentados, lembrando que eles participaram da
seleção da 3.9. Confira `sampling_mass_by_dataset_class` para ver quanto o
sampler realmente sorteou de cada fonte, além da contagem bruta de imagens.
Analise erros por segmento apenas quando houver amostras suficientes;
associações de formato, resolução ou categoria não comprovam causalidade.

### Como fazer a revisão visual da célula #10

A célula mostra até **seis falsos positivos (FP)** com maior probabilidade
de sintética e **seis falsos negativos (FN)** com menor probabilidade de
sintética. São os erros mais confiantes, selecionados de propósito: a galeria
não representa a frequência dos problemas em todas as imagens. Use a saída
numérica da célula #11 para contar erros por fonte e classe.

Para cada imagem, anote `FP` ou `FN`, fonte/grupo e `p` exibidos no título.
Observe se há padrões de conteúdo (por exemplo, retrato, objeto isolado,
paisagem ou cena complexa), textura, granulação, compressão, texto, bordas,
cores e escala. Em FP, pergunte que característica de uma foto real pode ter
sido confundida com síntese. Em FN, pergunte se a imagem sintética parece
fotográfica após a normalização JPEG ou se o recorte de 224 px pode ocultar
artefatos. Marque observações como **hipóteses visuais**; aparência não
confirma a origem nem corrige o rótulo.

Se uma imagem TGIF parecer rotulada incorretamente, use o `pair_id` em
`predictions_selection_paired.csv` e confira o par original/sintético no
manifesto TGIF congelado antes de concluir que há erro de rótulo. Não mude
o manifesto com base apenas na galeria. Não escolha exemplos, parâmetros
ou limiares a partir do Mirage.

### Leitura da frequência, após a auditoria executada

O resumo informado pelo grupo registra **59 FP e 182 FN** na seleção da
3.9. As sintéticas TGIF respondem por **115 dos 182 FN (63,2%)** e por
**115 dos 241 erros totais (47,7%)**; apenas 13/128 foram reconhecidas.
Nas 1.420 imagens comuns, a 3.9 acertou 1.308 e a 3.8, 1.303: não houve
queda relevante nesse subconjunto. A galeria de erros extremos mostrou
cinco FP reais AIGI e um Community Forensics, mas isso não significa que
AIGI concentre a maioria dos FP. A célula #11 mediu essa frequência real.

O sampler distribuiu **7,835% dos sorteios a TGIF real** e **1,740% a
TGIF sintética**, embora haja 512 imagens de cada classe no treino TGIF.
Essa desigualdade é uma condição do treino verificada por cálculo; seu
efeito causal precisa ser testado em uma nova variante com apenas o sampler
alterado. Não transformar o limiar ou a temperatura em correção baseada
nos erros externos observados.

Depois da execução, compartilhe o `summary.json` mostrado na célula #8 e
os segmentos de erro impressos abaixo dele. Se executar a #10, informe apenas
observações visuais pertinentes, sem usar essas imagens para ajustar a fase
concluída. Isso permite escolher uma única
hipótese para a próxima versão com base em evidência interna. Para demonstrar
a meta de 85% em um modelo futuro, será necessário ainda um teste final
independente, definido antes de observar suas predições.
