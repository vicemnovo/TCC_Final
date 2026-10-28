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
| `docs/BENCHMARK_EXTERNO_2000.md` | Novo protocolo de avaliação congelada RAISE + MNW, ainda sem resultados |
| `Notebooks/fase_3/avaliar_fases_benchmark_2000_colab.ipynb` | Piloto, preparo, cópia local e reavaliação manual das Fases 1–3.4 no mesmo conjunto |
| `ESTADO_PROJETO.md` | Estado verificado e próximos passos |
| `modelos/` | Checkpoints e resultados locais, fora do Git |
| `minhas_imagens/`, `validacao_fase3/` | Material externo legado, separado do treino e da calibração; `teste_externo/` local foi retirado após conferir cópias idênticas |
| `docs/historico/external_predictions_20_imagens.csv` | Predições legadas preservadas da pasta retirada |

Os dados, checkpoints e resultados no Drive permanecem sob `/content/drive/MyDrive/TCC_Final` com os caminhos existentes. O código sincronizado do GitHub é extraído no runtime temporário em `/content/tcc_codigo/<sha>`; alterações locais só passam a valer no Colab depois de publicadas na branch `fase3-ajuste-fino` com autorização.

Para continuar a Fase 3.4, abra [o guia de execução](docs/README_fase3_4.md) e depois [o notebook do Colab](Notebooks/fase_3/rodar_fase3_4_colab.ipynb). A execução salva da Fase 3.4 já contém resultados; não reinicie a pasta `fase3_4_finetune_v1`. As 20 imagens externas já avaliadas são históricas e não devem orientar treino, calibração ou seleção de hiperparâmetros. Para novas avaliações, preparar um conjunto congelado com alvo de 1.000 reais e 1.000 sintéticas no Drive e copiá-lo para `/content` em cada sessão, conforme `AGENTS.md`; a seleção ainda não existe.

A v2 tem [guia próprio](docs/README_fase3_4_v2.md) e [notebook próprio](Notebooks/fase_3/rodar_fase3_4_v2_colab.ipynb). Seu treino e sua [avaliação externa congelada](docs/AVALIACAO_EXTERNA_FASE3_4_V2.md) terminaram conforme saídas fornecidas pelo usuário; ela não substitui a v1. O resultado externo foi 11/20 acertos (55%), abaixo da meta de 85%. Não reutilizar estas 20 imagens para ajustar um novo modelo.

As Fases 1–3.3 são históricas: o notebook da Fase 1 está em [Notebooks/fase_1/](Notebooks/fase_1/README.md), e seu checkpoint local em `modelos/phase1_best_model.pt`. A futura API web ainda não existe no repositório e deve consumir um modelo congelado, sem acoplar treino à aplicação.
