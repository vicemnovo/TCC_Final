# Estado do projeto — Fase 3.4

## Trabalho atual

- Projeto: detector de imagens sintéticas com ViT-Tiny, FFT e EfficientNet-B0.
- Fase em desenvolvimento: Fase 3.4, ajuste fino isolado (`fase3_4_finetune_v1`).
- Branch local: `fase3-ajuste-fino`, acompanhando `origin/fase3-ajuste-fino`. Último commit registrado nesta cópia: `a54385b` (`Corrige preparacao da Fase 3.4 no Colab`).
- Código principal: `experimentos/fase3_4_finetune.py`. Interface de execução: `experimentos/rodar_fase3_4_colab.ipynb`. A arquitetura e funções da Fase 3.3 são reutilizadas de `Notebooks/fase3 - openfake/fase3.3_compressao.ipynb` no snapshot sincronizado do GitHub.

## Arquivos e armazenamento

- Arquivos do experimento existentes: `experimentos/fase3_4_finetune.py`, `experimentos/rodar_fase3_4_colab.ipynb`, `experimentos/configurar_github_colab.ipynb` e `experimentos/README_fase3_4.md`.
- Alterações locais ainda não publicadas na verificação deste documento: `AGENTS.md`, `experimentos/README_fase3_4.md`, `experimentos/rodar_fase3_4_colab.ipynb` e `Notebooks/fase3 - openfake/fase3.3_compressao.ipynb`. Preservar e revisar esta última alteração separadamente; não incluí-la automaticamente em um commit da Fase 3.4.
- Checkpoint inicial configurado: `/content/drive/MyDrive/TCC_Final/modelos/fase3_3_compressao_v1/phase3_final.pt`. O script também usa o checkpoint da Fase 2 em `/content/drive/MyDrive/TCC_Final/modelos/fase2_gate_vit_fft_cnn/phase2_final.pt` para reconstruir a arquitetura antes de carregar os pesos da Fase 3.3. Ambos existem na cópia local em `modelos/`.
- Dados preparados, quando a coleta terminar: `/content/drive/MyDrive/TCC_Final/dados_fase3_4/manifest.json` e imagens referenciadas no manifesto.
- Novos artefatos, quando produzidos: `/content/drive/MyDrive/TCC_Final/modelos/fase3_4_finetune_v1/`. O script prevê `last.pt`, `best.pt`, `history.csv`, `experiment_config.json`, `phase3_final.pt`, `final_metrics.json`, `per_group.csv` e `test_predictions.csv`. O smoke test usa a pasta separada `modelos/fase3_4_finetune_v1_smoke/`.
- O código baixado do GitHub fica no runtime temporário em `/content/tcc_codigo/<sha>`. Dados, checkpoints e resultados ficam no Drive. `credenciais/`, `modelos/` e `dados_fase3_4/` estão ignorados pelo Git.

## Execução e resultados verificados

- As saídas salvas em `rodar_fase3_4_colab.ipynb` mostram GPU Tesla T4, Drive montado, credencial desbloqueada, código sincronizado no commit `a54385b` e verificação do script e checkpoint inicial. Isso não comprova que a coleta de dados tenha terminado.
- Não há saída concluída das células 6 (preparação), 7 (smoke test) ou 8 (treino) no notebook salvo. Não existe diretório `fase3_4*` na pasta `modelos/` desta cópia local. O estado atual do Drive remoto não foi inspecionado nesta verificação.
- Treinamento da Fase 3.4: **não comprovado como iniciado ou concluído**. Última época concluída: **não disponível**. Métricas da Fase 3.4: **não disponíveis**. Não afirmar que a meta externa de 85% foi alcançada.
- Referência anterior, não resultado da Fase 3.4: `modelos/fase3_3_compressao_v1/final_metrics.json` registra AUC `0.8040601900739176`, acurácia no limiar ajustado `0.7113507960965588`, acurácia balanceada `0.7076388595564942` e melhor época `15` no teste daquele experimento.

## Retomada no Google Colab

1. Abra `experimentos/rodar_fase3_4_colab.ipynb` pelo VS Code e conecte a GPU do Colab. A configuração inicial da credencial já foi usada nas saídas salvas; não a refaça sem necessidade.
2. Em uma sessão nova, execute as células 1 a 5 em ordem: dependências e GPU, montagem do Drive, senha de desbloqueio, sincronização do snapshot do GitHub e verificação dos checkpoints. Na sessão já conectada, execute somente as células cujas variáveis ou versões estejam faltando.
3. Execute a célula 6. Ela pula a coleta se `dados_fase3_4/manifest.json` existir; caso contrário, prepara o OpenFake. Aguarde a confirmação de conclusão. A versão local atual do notebook mostra logs em tempo real e avisos a cada 30 segundos sem saída; para ter a mesma versão após abrir em outro computador, publique as alterações locais autorizadas na branch.
4. Execute a célula 7 para o smoke test. Confirme `modelos/fase3_4_finetune_v1_smoke/smoke.json` antes de considerar o ambiente validado.
5. A célula 8 é **manual**. Somente após verificar dados e smoke test, descomente `executar_com_progresso('auto')` e execute quando houver autorização para treinar. O modo `auto` inicia uma pasta nova, retoma de `last.pt` ou encerra se `final_metrics.json` já existir. Consulte `history.csv` e `last.pt` no Drive para determinar a última época real; não deduza pelo notebook.

## Problemas conhecidos e prioridades

- Uma falha anterior na preparação vinha de `drive.mount()` dentro do subprocesso. A correção está no commit `a54385b`; caso a falha reapareça, confirme que a célula 4 sincronizou esse commit ou um posterior.
- A saída anterior de `subprocess.run` escondia o erro interno até o fim da célula. O notebook local atual exibe logs durante a execução e avisos periódicos. Um aviso de processo ativo não prova avanço da coleta; acompanhe os logs e o manifesto.
- Próximo passo prioritário: confirmar o término da célula 6 e o manifesto no Drive; depois executar o smoke test da célula 7. Só então decidir sobre o treino completo. As 20 imagens externas não devem entrar no treino, calibração ou seleção de hiperparâmetros.
- Antes de trocar de computador, revisar e publicar somente as alterações locais pretendidas. Não fazer commit ou push automaticamente; não incluir credenciais, datasets, checkpoints nem resultados grandes no Git.
