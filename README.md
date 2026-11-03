# TCC_Final

Detector de imagens reais e sintéticas com ViT-Tiny, FFT e EfficientNet-B0. O foco atual é melhorar a acurácia final em imagens externas independentes; vídeos e aplicação web/API ficam para trabalho futuro. A meta de 85% de acurácia ainda não foi demonstrada.

O detector calcula primeiro uma **probabilidade estimada de imagem sintética**. O rótulo binário é uma decisão posterior, obtida ao comparar essa probabilidade com o limiar congelado do modelo. `p(real) = 1 - p(sintética)` no problema binário atual. Essa estimativa não mede a veracidade do conteúdo retratado e ainda não foi validada como probabilidade calibrada para toda imagem enviada por usuários. A aplicação web/API ainda será desenvolvida e deverá apresentar a probabilidade junto do rótulo, da versão do modelo e do limiar usado.

## Estrutura

| Caminho | Finalidade |
| --- | --- |
| `Notebooks/fase_1/` | Notebook original e guia histórico da Fase 1 |
| `Notebooks/fase_2/` | Treino e avaliação históricos da Fase 2 |
| `Notebooks/fase_3/` | Experimentos 3.1–3.3 e interfaces Colab das Fases 3.4–3.6 |
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
| `experimentos/diagnostico_interno_fase3_5.py` | Diagnóstico interno manual do checkpoint final da 3.5, sem testes externos |
| `experimentos/auditar_erros_internos_fase3_5.py` | Auditoria dos erros já registrados na seleção interna da 3.5 |
| `experimentos/fase3_6_escala.py`, `Notebooks/fase_3/rodar_fase3_6_colab.ipynb`, `docs/README_fase3_6.md` | Variante isolada de robustez a redimensionamento, treinada internamente no Colab |
| `experimentos/fase3_7_openimages_pilot.py`, `Notebooks/fase_3/piloto_fase3_7_openimages_colab.ipynb`, `docs/README_fase3_7_piloto.md` | Piloto manual de uma nova fonte real rastreável; sem treino ou seleção congelada |
| `experimentos/benchmark_mirage_2000.py`, `Notebooks/fase_3/avaliar_fase3_5_mirage_2000_colab.ipynb` | Seleção e avaliação manuais do Mirage-Test, concluídas para a Fase 3.5 conforme saída do Colab |
| `docs/BENCHMARK_EXTERNO_MIRAGE_2000.md` | Fonte, protocolo congelado, avaliação da 3.5 e limites do Mirage-Test já observado |
| `docs/GUIA_METODOLOGIA_TCC.md` | Arquitetura, probabilidade, partições, métricas e interpretação para apresentação do TCC |
| `docs/PLANO_ACURACIA_85.md` | Sequência prioritária para melhorar e avaliar a acurácia externa |
| `ESTADO_PROJETO.md` | Estado verificado e próximos passos |
| `modelos/` | Checkpoints locais legados, fora do Git; os checkpoints atuais ficam no Drive |
| `minhas_imagens/`, `validacao_fase3/` | Material externo legado, separado do treino e da calibração; `teste_externo/` local foi retirado após conferir cópias idênticas |
| `docs/historico/external_predictions_20_imagens.csv` | Predições legadas preservadas da pasta retirada |

Os dados, checkpoints e resultados no Drive permanecem sob `/content/drive/MyDrive/TCC_Final` com os caminhos existentes. O código sincronizado do GitHub é extraído no runtime temporário em `/content/tcc_codigo/<sha>`; alterações locais só passam a valer no Colab depois de publicadas na branch `fase3-ajuste-fino` com autorização.

As Fases 3.4 v1/v2 já têm resultados. As 20 imagens externas anteriores são históricas e não devem orientar treino, calibração ou seleção de hiperparâmetros. O benchmark RAISE/MNW de 1.000 reais e 1.000 sintéticas também foi avaliado nas sete fases anteriores; agora é diagnóstico histórico. Consulte [ESTADO_PROJETO.md](ESTADO_PROJETO.md) para a retomada exata e o [guia de metodologia](docs/GUIA_METODOLOGIA_TCC.md) para entender a arquitetura, as probabilidades e os indicadores.

A v2 tem [guia próprio](docs/README_fase3_4_v2.md) e [notebook próprio](Notebooks/fase_3/rodar_fase3_4_v2_colab.ipynb). Seu treino e sua [avaliação externa congelada](docs/AVALIACAO_EXTERNA_FASE3_4_V2.md) terminaram conforme saídas fornecidas pelo usuário; ela não substitui a v1. O resultado externo foi 11/20 acertos (55%), abaixo da meta de 85%. Não reutilizar estas 20 imagens para ajustar um novo modelo.

As sete fases anteriores foram avaliadas no benchmark RAISE/MNW congelado de 2.000 imagens, conforme saídas do Colab; a melhor acurácia foi 62,1% (3.4 v2). A [Fase 3.5](docs/README_fase3_5.md) obteve AUC de seleção interna 0,9488; a [Fase 3.6](docs/README_fase3_6.md), com mais redimensionamento, obteve 0,9472 nas mesmas partições. A 3.5 foi avaliada no [Mirage-Test 2000 v1](docs/BENCHMARK_EXTERNO_MIRAGE_2000.md) congelado e obteve **60,45% de acurácia externa**, conforme saída do Colab. A 3.6 ainda não tem avaliação externa. O Mirage-Test agora é diagnóstico visto e não deve selecionar a próxima variante ou servir de teste final independente dela.

As Fases 1–3.3 são históricas: o notebook da Fase 1 está em [Notebooks/fase_1/](Notebooks/fase_1/README.md), e seu checkpoint local em `modelos/phase1_best_model.pt`. A futura API web ainda não existe no repositório e deve consumir um modelo congelado, sem acoplar treino à aplicação.
