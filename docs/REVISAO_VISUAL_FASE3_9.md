# Revisão visual interna da Fase 3.9 — 19 pares exibidos

## Evidência disponível

Em 10/11/2026, o grupo executou
`Notebooks/fase_3/fase3_9/conferir_fase3_9_colab.ipynb`. A saída da célula 2
confirmou **1.536 registros e 768 pares TGIF**, com 512 pares no treino,
128 na seleção e 128 na calibração. O relatório V2 mais recente também
registrou 768 pares. O comprovante foi gravado no Drive como
`resultados/validacao_perguntas_3_9_v1/conferencia_pares_tgif_20261110_213426_837279_UTC.json`.

O `resumo.json` enviado pelo grupo informa 19 pares internos nas galerias,
extraídos dos até 30 candidatos impressos no log V3. O relatório V3 contém
**71 candidatos dHash** no total, incluindo comparações com o teste reservado.
Nenhuma imagem do teste reservado foi exibida pelo novo notebook. Foram
inspecionadas `galeria_01.jpg` embutida no notebook local e as galerias 02 e
03 enviadas no ZIP `drive-download-20261110T220132Z-1-001.zip`. Assim, os
**19 pares exibidos** foram revisados; os candidatos não impressos no log
permanecem fora desta revisão visual. O arquivo `candidatos.csv` foi recebido
depois do ZIP e confirmou as divisões exatas. Hashes SHA-256 dos arquivos:

- `resumo.json`: `6349a0099e02e70fa76a46f77299a02072308de7650f3234c7682506cec082b9`;
- ZIP das galerias 02/03: `cf93f3bd4073483119d13e4552e502625567933c6c7bbea42f493dbea0024892`;
- `candidatos.csv`: `f20b77d1a4357431ea696fb78074ca87e3bd8ac75b975de09f0fefdcb2640ed1`.

## Inspeção das três galerias

| Par na galeria | Observação visual | Interpretação limitada |
| --- | --- | --- |
| 1, dHash 0, MAE 0,84 | Os dois lados mostram a mesma fotografia de uma tenista na quadra, com diferenças mínimas de processamento. | `validation_selection` × `validation_calibration`, ambas reais. Repetição visual entre seleção e calibração. |
| 2, dHash 0, MAE 1,25 | Os dois lados mostram a mesma fotografia de uma tenista em quadra externa, com diferenças mínimas. | `train` × `validation_calibration`, ambas reais. Repetição visual entre treino e calibração. |
| 3, dHash 2, MAE 17,09 | A composição da mesa e do celular é a mesma, mas o conteúdo da tela muda. | `train` × `validation_calibration`, ambas sintéticas. Possível derivação da mesma cena, sem duplicata integral. |
| 4–19 | As imagens dos pares são visualmente diferentes. | Nesses casos, o dHash baixo isolado parece um falso candidato. |

Os dois pares mais fortes têm identificadores recuperados da saída V3:

- Par 1: `aigi:PA_alpha:ROG0d31E.jpg` e
  `cfsmall:n/a:76f3fe8115632b13fa50e8827ae9c1689d82f6d7e5e42ec58a1f402771d799fa.jpg`.
- Par 2: `cfsmall:n/a:ef64e3e4901af12b93c3fbc034243b5f9990576dc58c422902bf5f4f14c71ec7.jpg`
  e `aigi:SD_2.1:W6e7uZfE.jpg`.

O CSV confirmou que os três pares relacionados envolvem **calibração**. A
leitura anterior do par 2 como treino × seleção estava incorreta; o CSV
mostra treino × calibração. Nos 19 pares exibidos, não foi identificado um
duplicado visual entre treino e `validation_selection`. Isso não prova a
ausência de outros pares entre essas divisões nos candidatos não exibidos.

Essas observações são feitas a partir de miniaturas de uma galeria, não dos
arquivos originais em resolução plena. Não se deve alterar o resultado
histórico da 3.9 nem atribuir um tamanho de viés às métricas apenas com esta
inspeção. Porém, a afirmação de que **não há repetição visual entre divisões
internas** já não é defensável. Os pares observados atingem a **calibração**,
que compartilha conteúdo visual com seleção e treino. Portanto, a
temperatura, o limiar e a métrica aparente dessa divisão devem ser
interpretados com cautela. Os pares observados, isoladamente, não demonstram
viés na AUC de seleção usada para escolher a época.

## Próxima ação

1. Registrar a Fase 3.9 como experimento concluído **com essa limitação de
   separação interna**, sem retreinar ou recalibrar o checkpoint histórico.
2. Para uma variante 3.9.1, formar grupos de imagens idênticas/derivadas
   **antes** da divisão em treino, seleção e calibração, mantendo cada grupo
   em uma única divisão. Definir a regra usando só dados internos. Mirage e
   outras imagens externas observadas não entram na escolha de parâmetros.

O status desta revisão permanece **parcial em relação aos 71 candidatos V3**:
apenas 19 candidatos internos impressos no log foram exibidos. O arquivo
`resumo.json` corretamente mantém `visual_review_complete=false`. Os achados
dos pares 1 e 2 já bastam para registrar repetição visual entre divisões
internas, sem precisar tratar todos os demais candidatos dHash como duplicatas.
