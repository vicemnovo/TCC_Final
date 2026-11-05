# Guia de metodologia e leitura dos resultados

Este guia explica o detector no estado atual. Para o ponto de parada, consulte
[`ESTADO_PROJETO.md`](../ESTADO_PROJETO.md); para reproduzir uma execução,
consulte o guia da fase correspondente. Os números externos aqui citados vieram
das saídas do Colab fornecidas pelo autor; o Drive não foi montado nesta revisão.

## Pergunta e saída do modelo

A tarefa binária é distinguir **imagem real (classe 0)** de **imagem sintética
ou alterada por IA (classe 1)**. O modelo produz dois *logits* (valores antes
da normalização). A avaliação transforma esses valores em `p_sintetica` por
`softmax(logits / T)[1]`, com temperatura `T` fixada na calibração interna.
No problema binário, `p_real = 1 - p_sintetica`. O rótulo operacional é
`sintética` quando `p_sintetica >= limiar`, e `real` caso contrário.

Exemplo didático: com limiar 0,445, `p_sintetica = 0,70` gera o rótulo
“sintética”; `p_sintetica = 0,30` gera “real”. A probabilidade estimada não
é prova de origem da imagem, não mede se a cena retratada é verdadeira e não
está comprovadamente calibrada para toda imagem futura enviada por usuários.
Uma foto real de uma montagem física continua sendo real quanto ao processo de
captura; uma imagem gerada que retrata fato plausível continua sintética.

### Frase apresentada por imagem

Toda nova avaliação individual informa `p_real`, `p_sintetica` e a classe
operacional pelo **limiar interno congelado**, junto a uma frase pré-definida.
A implementação compartilhada está em
[`linguagem_predicao.py`](../experimentos/linguagem_predicao.py). As faixas
de texto são: abaixo de 20% sintética, chance estimada alta de real; de 20%
a menos de 40%, mais chance estimada de real; de 40% a menos de 60%, chances
próximas; de 60% a menos de 80%, mais chance estimada de sintética; a partir
de 80%, chance estimada alta de sintética. Essas faixas servem **somente para
redigir a mensagem**. Não são limiares de treinamento ou calibração.

A frase pode indicar chances próximas e, ainda assim, a classe operacional
ser sintética. Por exemplo, `p_sintetica=40%` e limiar congelado de `31%`
produzem `p_real=60%`, texto de chances próximas e classe operacional
**sintética**. Mostrar os três valores evita que a mensagem esconda essa
diferença. Mesmo após calibração interna, 80% é uma estimativa do modelo,
não 80% de certeza sobre a origem real da imagem. Benchmarks grandes mantêm
as métricas agregadas e os CSVs numéricos; a frase pode ser acrescentada
como uma coluna por imagem, sem mudar resultados históricos.

## Arquitetura e processamento

A arquitetura histórica combina três ramos: **ViT-Tiny** (padrões visuais
globais), **FFT** (representação em frequência) e **EfficientNet-B0** (padrões
locais de convolução). A saída dos ramos é combinada por uma fusão com *gate*,
que aprende pesos para a decisão conjunta. Isso descreve a intenção de cada
ramo; as métricas disponíveis não isolam quanto cada ramo contribuiu.
O ramo FFT calcula `log1p(abs(fftshift(fft2(x))))` em cada recorte de 224 px.
Isso oferece um sinal de frequência, mas não comprova que o modelo identifica
um artefato específico. ViT e CNN também não constituem, por si, verificadores
de coerência física. A Fase 3.7 em diante tratará ruído/pixel, consistência
de cor/pigmentação, relações globais e frequências como hipóteses verificáveis
em dados internos, uma mudança por vez. O detector continua recebendo pixels,
sem exigir metadados de origem de quem envia a imagem.
Os scripts das Fases 3.5 e 3.6 remontam essa arquitetura a partir de
[`fase3.3_compressao.ipynb`](../Notebooks/fase_3/fase3.3_compressao.ipynb),
carregam estritamente o checkpoint inicial da 3.3 e treinam uma nova cópia.

No treino há recorte, inversão horizontal, desfoque leve, redimensionamento
seguido de ampliação e compressão JPEG aleatória. A 3.6 alterou somente a
probabilidade desse redimensionamento, de 0,15 para 0,50; dados, partições e
checkpoint inicial foram mantidos. Na avaliação Mirage da 3.5, cada imagem
gera **cinco recortes de 224 px**. O programa calcula logits por recorte,
obtém a **média dos logits** e só então aplica a temperatura e o `softmax`.
Se o lado curto fosse menor que 224 px, haveria ampliação bicúbica antes dos
recortes; o relatório registrou zero casos assim. Média de probabilidades
teria outro significado e não reproduziria esse protocolo.

## Separação de dados e decisões

| Papel | O que decide | Pode mudar os pesos? |
| --- | --- | --- |
| Treino interno | Ajuste dos parâmetros por gradientes | Sim |
| Seleção interna | Melhor época/checkpoint pela AUC | Não diretamente |
| Calibração interna | Temperatura e limiar após selecionar o checkpoint | Não |
| Teste externo congelado | Medição de generalização com tudo fixado | Não |

Na 3.5, as três partições internas têm **3.594 / 719 / 719** imagens.
O teste OpenFake histórico de **1.947** imagens foi conferido quanto à
separação, sem inferência na 3.5. Os novos arquivos selecionados do Community
Forensics–Small foram baixados integralmente, persistidos no Drive e copiados
para `/content` antes do processamento, conforme a
[`política de dados`](POLITICA_DADOS_FUTUROS.md). A fonte das fotos reais desse
recorte aparece como `n/a`; a divisão delas é por imagem, e não há prova de
independência por fonte. A seleção interna mostrou associação entre classe e
resolução processada. Isso é uma limitação dos dados, não demonstra sozinho
qual pista o modelo usou.

As imagens RAISE/MNW, as 20 imagens antigas e o Mirage-Test já produziram
feedback. Seus resultados podem ser relatados como histórico, mas suas
predições, erros e grupos não podem definir dados, pesos, limiar ou
hiperparâmetros de uma variante futura. Um novo teste final independente
precisa ser definido e congelado antes de inferir a próxima versão. Hashes
iguais revelam cópias exatas; hashes diferentes não provam origens distintas.

## Como interpretar os indicadores

Com a classe sintética como positiva: **TP** é sintética corretamente
detectada; **FN** é sintética classificada como real; **TN** é real
corretamente reconhecida; **FP** é real classificada como sintética.

| Indicador | Cálculo | Pergunta respondida |
| --- | --- | --- |
| Acurácia | `(TP + TN) / N` | Que fração de todas as imagens foi acertada? |
| Recall sintético | `TP / (TP + FN)` | Quantas sintéticas foram encontradas? |
| Especificidade real | `TN / (TN + FP)` | Quantas reais foram preservadas como reais? |
| Precisão | `TP / (TP + FP)` | Das marcadas sintéticas, quantas eram sintéticas? |
| F1 | média harmônica de precisão e recall | Como se equilibram as duas medidas da classe sintética? |
| Acurácia balanceada | `(recall + especificidade) / 2` | Qual é a média do acerto por classe? |
| AUC ROC | área sob a curva de sensibilidade × taxa de FP | Quão bem a pontuação ordena reais e sintéticas em vários limiares? |

A **AUC** mede ordenação de pontuações em um conjunto; a **acurácia** depende
de um limiar específico. É possível ter AUC acima de 0,5 e acurácia próxima
de 0,5 quando o limiar congelado produz poucas detecções sintéticas.
**Temperatura não é meta de qualidade:** `T < 1` torna o softmax mais
concentrado, `T > 1` mais disperso; ela é estimada nos dados internos.
**Limiar também não deve ficar “mais perto de 1” para ser bom:** aumentá-lo
costuma reduzir FP e também reduzir recall. O valor adequado depende do
objetivo e deve ser escolhido exclusivamente na calibração interna.

## Evidência disponível e seus limites

| Avaliação | Modelo | Acurácia | AUC | Recall sintético | Nota |
| --- | --- | ---: | ---: | ---: | --- |
| Seleção interna, cinco recortes | 3.5 | 90,68% | 0,9696 | 89,59% | Partição usada para escolher o modelo; descritiva |
| Mirage-Test 2000 v1 | 3.5 | 60,45% | 0,6800 | 35,1% | Externo congelado, já observado |
| RAISE/MNW 2000 v1 | 3.4 v2 | 62,1% | 0,7375 | 28,9% | Outro modelo e outro conjunto; histórico |

No Mirage-Test, a matriz da 3.5 foi `TN=858, FP=142, FN=649, TP=351`.
A acurácia é `(858 + 351) / 2.000 = 0,6045`. O intervalo de Wilson de 95%
informado foi **58,29%–62,57%**. Esse intervalo expressa incerteza amostral
sob as hipóteses usuais; não inclui possível viés de seleção ou dependência
entre imagens. A diferença entre a seleção interna e Mirage é evidência de
generalização insuficiente nesse benchmark, mas não identifica uma causa
isolada. A meta externa de **85% ainda não foi alcançada**.

## Onde conferir a implementação

- [`fase3_5_finetune.py`](../experimentos/fase3_5_finetune.py): dados internos,
  verificação de hashes, seleção, calibração e checkpoints da 3.5.
- [`fase3_6_escala.py`](../experimentos/fase3_6_escala.py): variante com mudança
  única de `RandomDownUp`.
- [`benchmark_mirage_2000.py`](../experimentos/benchmark_mirage_2000.py):
  seleção fixa, cópia local, inferência de cinco recortes e cálculo das métricas.
- [`BENCHMARK_EXTERNO_MIRAGE_2000.md`](BENCHMARK_EXTERNO_MIRAGE_2000.md):
  origem, protocolo e resultado informados pelo Colab.

Os scripts e a extração de arquitetura usados na avaliação têm hashes
registrados no resultado. Para preservar a reprodutibilidade, explicações
posteriores ficam neste guia; qualquer mudança funcional futura deve usar
novo nome e diretório de experimento.
