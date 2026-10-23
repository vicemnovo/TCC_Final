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
| `docs/README_fase3_4.md` | Preparação, execução e retomada no Colab |
| `ESTADO_PROJETO.md` | Estado verificado e próximos passos |
| `modelos/` | Checkpoints e resultados locais, fora do Git |
| `minhas_imagens/`, `teste_externo/`, `validacao_fase3/` | Material externo legado, separado do treino e da calibração |

Os dados, checkpoints e resultados no Drive permanecem sob `/content/drive/MyDrive/TCC_Final` com os caminhos existentes. O código sincronizado do GitHub é extraído no runtime temporário em `/content/tcc_codigo/<sha>`; alterações locais só passam a valer no Colab depois de publicadas na branch `fase3-ajuste-fino` com autorização.

Para continuar a Fase 3.4, abra [o guia de execução](docs/README_fase3_4.md) e depois [o notebook do Colab](Notebooks/fase_3/rodar_fase3_4_colab.ipynb). A execução salva da Fase 3.4 já contém resultados; não reinicie a pasta `fase3_4_finetune_v1`. As 20 imagens externas são reservadas para avaliação independente e não devem orientar treino, calibração ou seleção de hiperparâmetros.

As Fases 1–3.3 são históricas: o notebook da Fase 1 está em [Notebooks/fase_1/](Notebooks/fase_1/README.md), e seu checkpoint local em `modelos/phase1_best_model.pt`. A futura API web ainda não existe no repositório e deve consumir um modelo congelado, sem acoplar treino à aplicação.
