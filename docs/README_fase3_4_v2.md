# Fase 3.4 v2 — amostragem menos repetitiva

A v1 continua intacta em `modelos/fase3_4_finetune_v1/`. A v2 usa o mesmo modelo ViT-Tiny + FFT + EfficientNet-B0, o mesmo checkpoint inicial da Fase 3.3, as mesmas épocas, taxas de aprendizado, aumentos, validação e calibração. A mudança experimental é a amostragem: classes continuam 50/50, mas a massa de cada gerador/fonte é proporcional à raiz quadrada da quantidade de imagens no treino, em vez de igual para todos os grupos. Isso reduz a repetição dos geradores com uma ou poucas imagens sem abandonar a diversidade de grupos.

O manifesto existente `dados_fase3_4/manifest.json` é reutilizado. A v2 **não** prepara, baixa ou altera dados. O replay em `/content/data/faces` é desativado porque essa pasta é efêmera e sua seleção não foi registrada na v1; portanto, uma comparação v1–v2 não isola perfeitamente apenas o sampler se a v1 tiver usado replay. A v2 registra `split_manifest.json`, o hash do manifesto, o número de imagens de replay (zero), a configuração e os hashes do script/fonte. Na retomada, recusa alterações nesses itens. Nenhum arquivo de teste é usado para escolher o sampler ou hiperparâmetros.

## Antes do Colab

Publique os novos arquivos na branch `fase3-ajuste-fino` **somente com autorização**. Alterações locais no VS Code não chegam ao runtime remoto até o push. Não repita a configuração das credenciais já concluída e não execute novamente o preparo de dados da v1.

## Execução no Colab

1. Abra `Notebooks/fase_3/rodar_fase3_4_v2_colab.ipynb` e conecte uma GPU.
2. Execute as células **1 a 6** em ordem. A célula 6 somente confirma o manifesto existente. A célula 4 baixa o snapshot publicado da branch para `/content/tcc_codigo/<sha>`.
3. Execute a célula **7** para o smoke test: partições, hashes, arquitetura, checkpoint, forward e gravação em `modelos/fase3_4_finetune_v2_sqrt_sampler_smoke/`. Não treina.
4. Confira o smoke. Para **iniciar ou retomar o treino manualmente**, execute a célula **8**, que chama `executar_com_progresso('auto')`. Se `final_metrics.json` já existir, não treina novamente. A célula **9** lê os artefatos sem alterar a avaliação.

O treino salva em `/content/drive/MyDrive/TCC_Final/modelos/fase3_4_finetune_v2_sqrt_sampler/`: `experiment_config.json`, `split_manifest.json`, `last.pt`, `best.pt`, `history.csv`, `phase3_final.pt`, métricas e predições finais. `last.pt` permite retomar após uma época concluída; se a sessão cair antes de fechar a primeira época, `auto` só reinicializa se houver apenas configuração/partições e nenhum checkpoint ou histórico parcial. A retomada preserva época, pesos e otimizador, mas não reproduz bit a bit a ordem aleatória dos lotes. Não apague arquivos para forçar uma retomada.

Durante o treino, acompanhe época, perda, AUC de validação, acurácia balanceada, melhor época e avanço dos checkpoints. A avaliação do teste OpenFake ocorre apenas após seleção do modelo e calibração pela validação interna; as 20 imagens externas permanecem fora de treino, calibração e seleção. Não há resultado da v2 até que o treino seja executado.
