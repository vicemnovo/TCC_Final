# Estado do projeto — Fase 3.4

## Trabalho atual

- Projeto: detector de imagens sintéticas com ViT-Tiny, FFT e EfficientNet-B0.
- Fase atual: Fase 3.4. A execução isolada `fase3_4_finetune_v1` consta como concluída na saída salva do notebook; a meta externa do projeto continua em aberto.
- Branch local: `fase3-ajuste-fino`, acompanhando `origin/fase3-ajuste-fino`. O snapshot registrado na execução do Colab foi `8fec4fff64cc064be5ae4c7892ba69137976ce70`.
- Código principal: `experimentos/fase3_4_finetune.py`. Interface de execução: `Notebooks/fase_3/rodar_fase3_4_colab.ipynb`. A arquitetura e funções da Fase 3.3 são reutilizadas de `Notebooks/fase_3/fase3.3_compressao.ipynb` no snapshot sincronizado do GitHub.
- O notebook histórico da Fase 1, `Notebooks/fase_1/fase1_vit.ipynb`, foi incorporado da cópia fornecida pelo autor. Somente a orientação inicial foi atualizada; o treino, as saídas salvas e o checkpoint existente não foram reexecutados ou alterados. Veja `Notebooks/fase_1/README.md`.

## Próxima execução: Fase 3.4 v2

- Código novo: `experimentos/fase3_4_finetune_v2.py`, `experimentos/amostragem_fase3_4.py` e `experimentos/particoes_fase3_4.py`; interface: `Notebooks/fase_3/rodar_fase3_4_v2_colab.ipynb`; instruções: `docs/README_fase3_4_v2.md`.
- Estado: **preparada localmente, sem treino ou smoke no Colab verificados**. A v1 permanece concluída e inalterada.
- Inicialização: checkpoint da Fase 3.3 em `/content/drive/MyDrive/TCC_Final/modelos/fase3_3_compressao_v1/phase3_final.pt`. Novos artefatos: `modelos/fase3_4_finetune_v2_sqrt_sampler/`; smoke separado em `modelos/fase3_4_finetune_v2_sqrt_sampler_smoke/`.
- A mudança experimental é a amostragem com massa por grupo proporcional a `sqrt(n)` e classes 50/50. O manifesto da v1 é reutilizado; replay efêmero desativado para garantir retomada consistente. Como a seleção real de replay da v1 não foi persistida, a comparação causal entre v1 e v2 tem essa limitação.
- O novo `split_manifest.json` registra as partições efetivas. A retomada compara manifesto, partições, configuração e hashes de código; não reavalia nem modifica a v1. Depois de publicar com autorização, executar células 1–7 do notebook novo e, somente após verificar o smoke, a célula 8 manual para treinar.
- Verificações locais da v2: três testes unitários sem GPU passaram (balanceamento de classes, redução de repetição de grupo raro, persistência/recusa de partições alteradas); sintaxe dos scripts e das células Python do notebook conferida. Ainda faltam smoke e importação com dependências reais na GPU do Colab.
- Auditoria recebida da célula 11 da v1: três pares de duplicatas exatas estão **somente no teste**, todos em `seedream-v5.0`; não há duplicata exata treino/teste. Doze pares treino/teste foram sinalizados por dHash, mas a MAE RGB não confirma identidade; nenhuma imagem foi removida. Esses dados servem para integridade, não para ajustar hiperparâmetros da v2.

## Arquivos e armazenamento

- Arquivos do experimento: `experimentos/fase3_4_finetune.py`, `Notebooks/fase_3/rodar_fase3_4_colab.ipynb`, `Notebooks/configuracao/configurar_github_colab.ipynb`, `Notebooks/configuracao/configurar_hf_colab.ipynb` e `docs/README_fase3_4.md`.
- Reorganização: cinco notebooks históricos passaram para `Notebooks/fase_2/` e `Notebooks/fase_3/`; três notebooks Colab passaram de `experimentos/` para `Notebooks/`; o guia passou para `docs/`. `README.md` documenta a árvore. As alterações anteriores já existentes no notebook da Fase 3.3, no script e na interface Colab foram preservadas.
- Checkpoint inicial configurado: `/content/drive/MyDrive/TCC_Final/modelos/fase3_3_compressao_v1/phase3_final.pt`. O script também usa o checkpoint da Fase 2 em `/content/drive/MyDrive/TCC_Final/modelos/fase2_gate_vit_fft_cnn/phase2_final.pt` para reconstruir a arquitetura antes de carregar os pesos da Fase 3.3. Ambos existem na cópia local em `modelos/`.
- Dados preparados no Drive segundo a saída salva no notebook: `/content/drive/MyDrive/TCC_Final/dados_fase3_4/manifest.json` e imagens referenciadas no manifesto.
- Artefatos da execução: `/content/drive/MyDrive/TCC_Final/modelos/fase3_4_finetune_v1/`. O script grava `last.pt`, `best.pt`, `history.csv`, `experiment_config.json`, `phase3_final.pt`, `final_metrics.json`, `per_group.csv` e `test_predictions.csv`. O smoke test usa a pasta separada `modelos/fase3_4_finetune_v1_smoke/`. A existência atual de cada arquivo no Drive não foi conferida diretamente nesta sessão local.
- O código baixado do GitHub fica no runtime temporário em `/content/tcc_codigo/<sha>`. Dados, checkpoints e resultados ficam no Drive. `credenciais/`, `modelos/` e `dados_fase3_4/` estão ignorados pelo Git.

## Execução e resultados verificados

- As saídas salvas em `rodar_fase3_4_colab.ipynb` mostram GPU Tesla T4, código sincronizado no commit `8fec4fff64cc064be5ae4c7892ba69137976ce70`, manifesto preservado no Drive e smoke test concluído com `Smoke OK`. O código do smoke carrega estritamente os pesos da Fase 3.3 e executa um forward finito; não testa backward, memória do lote completo ou retomada.
- A célula manual de treino já possui saída `auto: concluído em 1.9 min`: épocas 1–4 concluídas, com early stopping após três épocas sem melhorar a AUC da validação. Melhor época: **1** (AUC de validação **0,9632**). Portanto, **não reiniciar este experimento**. O estado atual do Drive remoto não foi inspecionado diretamente nesta sessão local.
- Métricas finais registradas na saída do notebook para o teste OpenFake: acurácia **0,71546**, AUC **0,81647**, acurácia balanceada **0,71186**, precisão **0,77872**, recall **0,57973**, F1 **0,66465**, especificidade **0,844**; limiar calibrado **0,46**. Esses números não são a avaliação das 20 imagens externas independentes e não comprovam a meta de 85%.
- Referência anterior, não resultado da Fase 3.4: `modelos/fase3_3_compressao_v1/final_metrics.json` registra AUC `0.8040601900739176`, acurácia no limiar ajustado `0.7113507960965588`, acurácia balanceada `0.7076388595564942` e melhor época `15` no teste daquele experimento.

## Retomada no Google Colab

1. Abra `Notebooks/fase_3/rodar_fase3_4_colab.ipynb` pelo VS Code e conecte a GPU do Colab. A configuração inicial da credencial já foi usada nas saídas salvas; não a refaça sem necessidade. Confirme que a branch remota contém a estrutura nova antes de sincronizar o snapshot.
2. Em uma sessão nova, execute as células 1 a 5 em ordem: dependências e GPU, montagem do Drive, senha de desbloqueio, sincronização do snapshot do GitHub e verificação dos checkpoints. Na sessão já conectada, execute somente as células cujas variáveis ou versões estejam faltando.
3. Confira no Drive `modelos/fase3_4_finetune_v1/final_metrics.json`, `history.csv`, `best.pt` e `last.pt`. A saída salva no notebook indica conclusão, mas esta cópia local não monta o Drive. Não execute novamente as células de preparação ou treino sem necessidade.
4. Para ler as métricas em nova sessão, rode as células 1–5 e depois a célula 9 de leitura de resultados. A célula 8 contém `executar_com_progresso('auto')` e é **manual**; se `final_metrics.json` existir, o modo `auto` apenas informa que o experimento está concluído.

## Problemas conhecidos e prioridades

- Uma falha anterior na preparação vinha de `drive.mount()` dentro do subprocesso. A correção está no commit `a54385b`; caso a falha reapareça, confirme que a célula 4 sincronizou esse commit ou um posterior.
- Após a reorganização, a célula 39 do notebook histórico da Fase 3.3 ainda consulta cópias antigas em `MyDrive/TCC_Final/Notebooks/fase3 - openfake/`. Ela é exploratória e não é executada pelo script da Fase 3.4; os caminhos do Drive foram preservados deliberadamente. A existência dessas cópias no Drive não foi verificada nesta sessão.
- O `.gitignore` bloqueia novos dados, modelos e credenciais, mas 62 arquivos de imagens/CSV externos já constam do índice Git. A reorganização não os removeu nem desindexou; um saneamento do histórico/publicação exige decisão separada.
- A validação local confirmou JSON de nove notebooks, sintaxe do script, índices de células e referências do Colab. Não há `torch`, `torchvision`, `datasets`, `sklearn` nem `cryptography` no Python local; importação funcional, GPU e sincronização remota dependem de teste no Colab com a estrutura nova.
- A saída anterior de `subprocess.run` escondia o erro interno até o fim da célula. O notebook local atual exibe logs durante a execução e avisos periódicos. Um aviso de processo ativo não prova avanço da coleta; acompanhe os logs e o manifesto.
- Próximo passo prioritário: confirmar diretamente no Drive os artefatos finais e analisar a diferença entre AUC de validação (0,9632) e AUC de teste OpenFake (0,81647), sem usar o teste nem as 20 imagens externas para ajustar hiperparâmetros deste mesmo experimento. A retomada após desconexão foi implementada por `last.pt`, mas não há registro de uma desconexão/retomada efetiva nesta execução.
- Antes de trocar de computador, confira que a branch local e `origin/fase3-ajuste-fino` apontam para o mesmo commit. Publicações futuras exigem autorização e não devem incluir credenciais, datasets, checkpoints nem resultados grandes no Git.
