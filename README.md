# TCC_Final — detecção de imagens reais e sintéticas

Projeto de TCC em Python/PyTorch. O modelo combina **ViT-Tiny**, uma
representação de **frequência por FFT** e **EfficientNet-B0** para estimar a
probabilidade de uma imagem ser sintética. O limiar do checkpoint transforma
essa probabilidade em uma decisão real/sintética. A probabilidade é uma
estimativa do modelo; não é prova da origem da imagem nem da veracidade da cena.

O objetivo experimental é atingir **pelo menos 85% de acurácia em um teste
externo independente**. Ainda não há evidência de que a meta foi alcançada.
Vídeos e aplicação web/API estão fora da etapa atual; a prioridade é melhorar
a generalização do detector de imagens.

## Ponto de partida

**Fase 3.8:** 480 pares AIGI foram preparados; o treino interno e a avaliação
comparativa Mirage foram concluídos conforme saídas do Colab. A acurácia foi
59,9%, abaixo dos 62,3% da 3.7 nas mesmas imagens. Consulte
[`ESTADO_PROJETO.md`](ESTADO_PROJETO.md) antes de continuar; ele contém as
células executadas, os limites dos resultados e os pilotos da Fase 3.9.
O procedimento histórico da 3.8 está em
[`docs/README_fase3_8_execucao.md`](docs/README_fase3_8_execucao.md).
Os pilotos combinados de Open Images e pares AIGI, ainda sem treino, estão em
[`docs/README_fase3_9.md`](docs/README_fase3_9.md).
As provas AIGI ampliadas tiveram baixa disponibilidade de fotos reais. O
[piloto TGIF](docs/README_fase3_9_tgif.md) confirmou 341 IDs pareados em
`validation`. A Fase 3.9 concluiu preparo, smoke, treino interno e
[avaliação comparativa Mirage](docs/README_fase3_9_execucao.md) com pesos e
limiar congelados: acurácia **58,75%** e AUC **0,6328**. A auditoria interna
da 3.9 foi concluída; a próxima tarefa é definir uma alteração controlada
para a **3.9.1**. Consulte o
[guia de retomada](docs/GUIA_RETOMADA_FASE3_9.md).
O [notebook de auditoria interna da 3.9](Notebooks/fase_3/auditar_interno_fase3_9_colab.ipynb)
e seu [relatório de resultados e limites](docs/AUDITORIA_INTERNA_FASE3_9.md)
documentam 59 FP e 182 FN nas 1.676 imagens da seleção interna.
O [notebook final da 3.9](Notebooks/fase_3/fase3_9_final_colab.ipynb)
reúne a preparação necessária do notebook de execução com as perguntas
V1–V3 do notebook de validação. O
[guia de fechamento](docs/README_fase3_9_fechamento.md) descreve as células
pendentes, os relatórios com data/hora e os limites da avaliação Mirage.
Para conferir os 768 pares TGIF e gerar galerias dos candidatos visuais em
um arquivo fácil de localizar, abra o
[notebook de conferências da 3.9](Notebooks/fase_3/conferir_fase3_9_colab.ipynb).
A revisão visual ainda depende de inspecionar as galerias; nenhum teste ou
treino é repetido por esse notebook.

O smoke test verifica dados, checkpoint e uma passagem pelo modelo; não mede
acurácia. O treino escolhe a melhor época na seleção interna e ajusta
temperatura/limiar apenas na calibração interna. A avaliação Mirage ocorre
depois do congelamento desses valores.

## Estrutura do repositório

| Pasta/arquivo | Conteúdo |
| --- | --- |
| `Notebooks/fase_1/` e `Notebooks/fase_2/` | Notebooks históricos das primeiras fases |
| `Notebooks/fase_3/` | Notebooks históricos e interfaces de execução no Colab |
| `Notebooks/configuracao/` | Configuração inicial das credenciais cifradas |
| `experimentos/` | Scripts Python principais de preparo, treino e avaliação |
| `tests/` | Verificações locais sem iniciar treino completo |
| `docs/` | Metodologia, resultados históricos, protocolos e guias |
| `ESTADO_PROJETO.md` | Estado verificado e próxima ação |
| `.gitignore` | Impede novos dados, credenciais e checkpoints no Git |

O código da branch `fase3-ajuste-fino` é copiado para
`/content/tcc_codigo/<sha>` no Colab. Datasets selecionados, manifestos,
checkpoints e resultados ficam sob `/content/drive/MyDrive/TCC_Final` no
Google Drive. A cópia em `/content` é temporária e deve ser refeita após uma
desconexão. O notebook funciona como interface; os scripts em `experimentos/`
contêm a implementação principal.

Os arquivos `.ipynb` e `.py` publicados na branch também podem ser espelhados
no Drive, nas mesmas pastas relativas, pelo
[notebook de sincronização](Notebooks/configuracao/sincronizar_codigo_drive_colab.ipynb).
Execute suas células 1–5 após cada atualização publicada no GitHub. O
[procedimento](docs/SINCRONIZACAO_CODIGO_DRIVE.md) explica a preservação de
versões anteriores e como conferir a cópia.

## Resultados e limites

| Conjunto | Modelos já avaliados | Melhor acurácia informada |
| --- | --- | ---: |
| RAISE/MNW 2000 | Fases 1–3.4 v2 | 62,1% (3.4 v2) |
| Mirage 2000 | Fases 3.5–3.9 | 62,3% (3.7); 3.8: 59,9%; 3.9: 58,75% |

Os dois conjuntos têm imagens e protocolos diferentes; não comparar suas
porcentagens como uma única série de desempenho. O Mirage já foi observado
durante o desenvolvimento: serve para **comparação descritiva entre fases**,
sem fornecer dados para treino, calibração ou escolha de hiperparâmetros.
Para demonstrar a meta em teste cego independente de um modelo desenvolvido
depois desses resultados, será necessário reservar outro conjunto externo.
Os resultados da antiga avaliação de 20 imagens continuam históricos.

## Guias para o grupo

- [Estado e retomada](ESTADO_PROJETO.md): o que já ocorreu e a próxima célula.
- [Guia de Git e Colab](docs/GUIA_GIT_E_COLAB.md): comandos básicos,
  publicação de código e cuidado com dados e outputs dos notebooks.
- [Metodologia](docs/GUIA_METODOLOGIA_TCC.md): arquitetura, probabilidade,
  divisões e significado das métricas.
- [Plano de acurácia](docs/PLANO_ACURACIA_85.md): prioridades após a 3.8.
- [Auditoria interna e opções 3.9](docs/AUDITORIA_E_OPCOES_FASE3_9.md):
  resultado interno da 3.8, proteção das divisões e alternativas de dados.
- [Pilotos combinados da 3.9](docs/README_fase3_9.md): passos manuais antes de
  congelar novos dados ou iniciar treino; histórico Open Images/AIGI.
- [Piloto TGIF/TGIF2 da 3.9](docs/README_fase3_9_tgif.md): validação de
  acesso, tamanhos e vínculo entre originais e imagens alteradas.
- [Avaliações históricas](docs/COMPARATIVO_RESULTADOS.md): Fases 1–3.4;
  [Mirage](docs/BENCHMARK_EXTERNO_MIRAGE_2000.md): protocolo das fases atuais.

Os modelos e resultados anteriores permanecem preservados. Não publicar
tokens, senhas, datasets ou checkpoints no GitHub.
