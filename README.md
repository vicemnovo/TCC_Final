# TCC_Final

Detector de imagens reais e sintéticas com ViT-Tiny, FFT e EfficientNet-B0. O foco atual é robustez em imagens; vídeos ficaram para trabalho futuro. A meta de 85% de acurácia no conjunto externo independente ainda não foi demonstrada.

## Estrutura

| Caminho | Finalidade |
| --- | --- |
| `Notebooks/fase_1/` | Notebook original e guia histórico da Fase 1 |
| `Notebooks/fase_2/` | Treino e avaliação históricos da Fase 2 |
| `Notebooks/fase_3/` | Experimentos 3.1–3.3 e interface Colab da 3.4 |
| `Notebooks/configuracao/` | Configuração inicial de credenciais GitHub/Hugging Face |
| `experimentos/fase3_4_finetune.py` | Implementação principal da Fase 3.4 |
| `experimentos/fase3_4_finetune_v2.py`, `experimentos/amostragem_fase3_4.py`, `experimentos/particoes_fase3_4.py` | Variante v2, amostragem e partições registradas |
| `tests/` | Testes locais sem GPU |
| `docs/README_fase3_4.md` | Preparação, execução e retomada no Colab |
| `docs/README_fase3_4_v2.md` | Experimento isolado de amostragem, concluído conforme saída do Colab |
| `docs/AVALIACAO_EXTERNA_FASE3_4_V2.md` | Procedimento manual e único para avaliar a v2 congelada nas 20 imagens externas |
| `docs/COMPARATIVO_RESULTADOS.md` | Resultados das Fases 1–3.4 v2 e critérios para decidir a etapa seguinte |
| `docs/POLITICA_DADOS_FUTUROS.md` | Preparação local de novos datasets no Colab; não altera o OpenFake nem a Fase 3.4 |
| `docs/BENCHMARK_EXTERNO_2000.md` | Benchmark RAISE + MNW congelado e avaliações históricas concluídas |
| `Notebooks/fase_3/avaliar_fases_benchmark_2000_colab.ipynb` | Piloto, preparo, cópia local e reavaliação manual das Fases 1–3.4 no mesmo conjunto |
| `docs/README_fase3_5.md`, `Notebooks/fase_3/rodar_fase3_5_colab.ipynb` | Piloto de novos dados internos e interface manual do ajuste fino 3.5 |
| `experimentos/fase3_5_dados.py`, `experimentos/fase3_5_finetune.py` | Preparo imutável dos dados e treino 3.5 isolado |
| `ESTADO_PROJETO.md` | Estado verificado e próximos passos |
| `modelos/` | Checkpoints e resultados locais, fora do Git |
| `minhas_imagens/`, `validacao_fase3/` | Material externo legado, separado do treino e da calibração; `teste_externo/` local foi retirado após conferir cópias idênticas |
| `docs/historico/external_predictions_20_imagens.csv` | Predições legadas preservadas da pasta retirada |

Os dados, checkpoints e resultados no Drive permanecem sob `/content/drive/MyDrive/TCC_Final` com os caminhos existentes. O código sincronizado do GitHub é extraído no runtime temporário em `/content/tcc_codigo/<sha>`; alterações locais só passam a valer no Colab depois de publicadas na branch `fase3-ajuste-fino` com autorização.

Para continuar a Fase 3.4, abra [o guia de execução](docs/README_fase3_4.md) e depois [o notebook do Colab](Notebooks/fase_3/rodar_fase3_4_colab.ipynb). A execução salva da Fase 3.4 já contém resultados; não reinicie a pasta `fase3_4_finetune_v1`. As 20 imagens externas já avaliadas são históricas e não devem orientar treino, calibração ou seleção de hiperparâmetros. O novo benchmark congelado de 1.000 reais e 1.000 sintéticas está no Drive, segundo saídas do Colab; ainda faltam suas avaliações. Em cada nova sessão, copie-o para `/content` e confira os hashes antes da inferência.

A primeira tentativa de avaliar a Fase 1 nesse benchmark encontrou uma falha de carregamento de classe antes de produzir métricas. O avaliador foi corrigido, mas ainda requer teste remoto. Consulte [ESTADO_PROJETO.md](ESTADO_PROJETO.md) para a retomada exata; não repita o reparo nem a preparação das imagens.

A v2 tem [guia próprio](docs/README_fase3_4_v2.md) e [notebook próprio](Notebooks/fase_3/rodar_fase3_4_v2_colab.ipynb). Seu treino e sua [avaliação externa congelada](docs/AVALIACAO_EXTERNA_FASE3_4_V2.md) terminaram conforme saídas fornecidas pelo usuário; ela não substitui a v1. O resultado externo foi 11/20 acertos (55%), abaixo da meta de 85%. Não reutilizar estas 20 imagens para ajustar um novo modelo.

As sete fases também foram avaliadas no benchmark RAISE/MNW congelado de 2.000 imagens, conforme saídas do Colab; a melhor acurácia foi 62,1% (3.4 v2). Esse conjunto agora é diagnóstico histórico, não teste final independente da próxima versão. A [Fase 3.5](docs/README_fase3_5.md) prepara um piloto de dados internos novos e um ajuste fino isolado; ainda não foi treinada nem avaliada.

As Fases 1–3.3 são históricas: o notebook da Fase 1 está em [Notebooks/fase_1/](Notebooks/fase_1/README.md), e seu checkpoint local em `modelos/phase1_best_model.pt`. A futura API web ainda não existe no repositório e deve consumir um modelo congelado, sem acoplar treino à aplicação.
