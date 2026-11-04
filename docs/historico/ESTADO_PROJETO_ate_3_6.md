# Estado do projeto — Fases 3.5 e 3.6 treinadas internamente

**Registro histórico.** Este arquivo preserva o estado anterior à execução
da 3.7 e ao preparo da 3.8. As indicações de "próxima atividade" abaixo
pertencem àquele momento. Para continuar agora, use
[`ESTADO_PROJETO.md`](../../ESTADO_PROJETO.md).

## Retomada rápida — atualizada em 03/11/2026

**Atualização de 03/11/2026:** a avaliação comparativa da 3.6 nas mesmas
2.000 imagens Mirage foi informada como concluída pelo usuário: acurácia
**0,572**, AUC **0,6303885**, recall sintético **0,288** e especificidade
real **0,856**. Na comparação pareada com a 3.5, ambas acertaram 1.123,
somente a 3.5 acertou 86, somente a 3.6 acertou 21 e ambas erraram 770.
O Drive não foi montado nesta sessão local. A próxima atividade é definir
uma variante 3.7 com novos dados internos variados; o piloto Open Images é
opcional, não uma condição para a inferência de imagens de usuários. Mirage será
o benchmark comparativo externo de cada fase futura, mas já é um conjunto
observado e não é teste cego independente para orientar desenvolvimento.
O roteiro atualizado em `docs/PLANO_ACURACIA_85.md` prioriza 3.7 com dados
reais/sintéticos variados e arquitetura preservada; 3.8 considera contexto
global e consistência de recortes se houver evidência interna; 3.9 considera
uma pista forense explícita (ruído/pixel, cor ou frequência) somente após
auditoria interna. Nenhuma dessas fases foi treinada ou avaliada ainda.

**Registro anterior:** a branch estava sincronizada e limpa no início
da retomada. Foi preparado localmente um piloto da possível Fase 3.7 com
Open Images, em `experimentos/fase3_7_openimages_pilot.py`,
`Notebooks/fase_3/piloto_fase3_7_openimages_colab.ipynb` e
`docs/README_fase3_7_piloto.md`. O piloto ainda **não foi executado no Colab**:
não há contagem verificada de candidatos, imagens baixadas, partições novas,
treino 3.7 ou métricas novas da 3.7. O piloto Open Images só deve ser
executado se a fonte for selecionada por critérios internos de diversidade
e qualidade; não é o próximo passo obrigatório. As Fases 3.5 e 3.6 e os resultados Mirage permanecem
como descritos abaixo; nenhum benchmark observado orienta a seleção do piloto.

- **Branch:** `fase3-ajuste-fino`. Em outro computador, confira o commit atual
  com `git rev-parse HEAD` e a sincronização com `git status -sb` antes de
  abrir notebooks. Os notebooks de execução baixam um snapshot publicado da
  branch; alterações só locais não chegam ao Colab.
- **Ponto de parada:** Fase 3.5 e variante 3.6 treinadas internamente. A 3.5
  já foi avaliada uma vez no Mirage-Test 2000 v1: **1.209/2.000 acertos,
  60,45%**, AUC **0,679972**, recall sintético **35,1%**. O Drive não foi
  montado nesta revisão; o resultado foi fornecido pela saída do Colab.
  A 3.6 obteve **1.144/2.000 acertos (57,2%)** na mesma avaliação
  comparativa, conforme saída do usuário. A meta de 85% segue não demonstrada.
- **Artefatos remotos a preservar:**
  `/content/drive/MyDrive/TCC_Final/modelos/fase3_5_cfsmall_v1/`,
  `/content/drive/MyDrive/TCC_Final/modelos/fase3_6_escala_p05_v1/` e
  `/content/drive/MyDrive/TCC_Final/resultados/benchmark_externo_mirage_2000_v1/fase3_5_cfsmall_v1/`.
  As partições internas persistidas da 3.5 ficam em
  `/content/drive/MyDrive/TCC_Final/dados_fase3_5/fase3_5_novos_v1/`.
- **Próxima atividade de pesquisa:** usar apenas a auditoria **interna**
  registrada na 3.5 para especificar uma variante isolada com imagens reais
  e sintéticas visualmente variadas e verificar os desequilíbrios de resolução
  e formato. A origem dos dados é útil para rótulos e separação quando existe;
  não será exigida de imagens enviadas por usuários. Fixar
  seleção, exclusões e métricas internas antes do novo treino; criar diretório
  e nome novos. Ver [`PLANO_ACURACIA_85.md`](../PLANO_ACURACIA_85.md).
- **Avaliação final futura:** definir e congelar outro conjunto externo ainda
  não observado. Mirage-Test, RAISE/MNW e as 20 imagens antigas já deram
  feedback e não podem escolher a próxima variante. Não repetir preparo,
  smoke ou treino concluídos das Fases 3.5/3.6 apenas para retomar a sessão.
- **Revisão local de encerramento:** documentação de metodologia e
  `.gitignore` ampliados, sem alterar scripts avaliados, datasets ou
  checkpoints. Os 33 testes locais passaram. Saídas de execução já salvas
  em notebooks são material do autor e devem ser preservadas.

## Trabalho atual

- Projeto: detector de imagens sintéticas com ViT-Tiny, FFT e EfficientNet-B0.
- Benchmark externo **congelado e avaliado**: 1.000 fotografias RAISE + 1.000 imagens MNW, 25 de cada um de 40 grupos elegíveis. A primeira coleta teve seis pares reais duplicados, substituídos antes do congelamento; a auditoria informou ausência de sobreposição exata com OpenFake v2 e as 20 imagens históricas. A cópia local verificou 2.000/2.000 imagens em `/content/benchmark_externo_raise_mnw_2000_v1`. O conjunto congelado fica em `/content/drive/MyDrive/TCC_Final/dados_externos/benchmark_externo_raise_mnw_2000_v1`. Todas essas verificações e as métricas abaixo vieram das saídas do Colab fornecidas pelo usuário; esta sessão local não montou o Drive.
- As sete avaliações históricas terminaram no mesmo manifesto SHA-256 `2024b501e2c409c5859ecfa4c13a50b233bf23acd7ee8aeeb6da3d868f328d53` e mesmo script SHA-256 `98b2d27c0a9689fb5a79e8bfec9f694c32a5930f61c0f1e2e381913ad6e38ab1`. A primeira tentativa da Fase 1 falhou por carregar a classe errada, mas a correção foi validada no Colab: o checkpoint foi carregado e 2.000 inferências concluíram. Resultados em `resultados/benchmark_externo_raise_mnw_2000_v1/<fase>/`, sem sobrescrever avaliações anteriores. O benchmark agora é diagnóstico visto; não usá-lo para escolher dados ou hiperparâmetros da 3.5 nem como teste final independente dela.
- Acurácia / AUC / recall sintético / especificidade: Fase 1 **0,502 / 0,5425 / 0,006 / 0,998**; Fase 2 **0,500 / 0,6426 / 0 / 1,000**; Fase 3.1 **0,5505 / 0,5568 / 0,317 / 0,784**; Fase 3.2 **0,5655 / 0,6048 / 0,288 / 0,843**; Fase 3.3 **0,5995 / 0,7384 / 0,226 / 0,973**; Fase 3.4 v1 **0,592 / 0,7362 / 0,207 / 0,977**; Fase 3.4 v2 **0,621 / 0,7375 / 0,289 / 0,953**. A melhor acurácia foi 62,1%, abaixo da meta experimental de 85%.
- **Fase 3.5 treinada internamente:** `experimentos/fase3_5_dados.py`, `experimentos/fase3_5_finetune.py`, `Notebooks/fase_3/rodar_fase3_5_colab.ipynb`, `docs/README_fase3_5.md` e testes. Fonte: Community Forensics–Small em shards Parquet completos, apenas para pesquisa não comercial. O piloto remoto concluiu: `HFCF_small_0` trouxe 2.939 sintéticas de 78 geradores (77 grupos com 20+), `HFCF_small_117` trouxe 2.992 reais, porém todas com `real_source=n/a`; 1.384.257.833 bytes baixados em `/content`, commit do dataset `6c539a534c07917307c381f5af4053c6091b5278`. A divisão das reais foi por imagem, não por fonte. O preparo congelou 3.000 imagens internas em `dados_fase3_5/fase3_5_novos_v1/`. O smoke passou com GPU Tesla T4, hashes das partições, pesos iniciais da Fase 3.3 e forward finito. O treino foi concluído em 12 épocas, com melhor checkpoint na época 10 por AUC de seleção **0,9487973**. A calibração interna retornou temperatura **0,4409471**, limiar **0,445** e acurácia balanceada aparente **0,9212768**. Partições: treino **3.594**, seleção **719**, calibração **719**; teste OpenFake histórico **1.947**, reservado e não avaliado. O usuário conferiu a presença de `experiment_config.json`, `split_manifest.json`, `history.csv`, `last.pt`, `best.pt`, `phase3_final.pt`, `internal_metrics.json` e `training_complete.json` em `modelos/fase3_5_cfsmall_v1/`. Esses fatos vêm das saídas do Colab; o Drive não foi montado nesta sessão local. A métrica externa Mirage-Test aparece abaixo, separada das métricas internas.
- O diagnóstico da seleção interna da 3.5 foi informado como concluído no Colab: 719 imagens, acurácia **0,9068150**, acurácia balanceada **0,9079695**, precisão **0,9313984**, recall sintético **0,8959391**, especificidade real **0,92**, F1 **0,9133247**, AUC **0,9696134** com cinco recortes; TN=299, FP=26, FN=41, TP=353. Foi usado o checkpoint SHA-256 `1dd78d4cd32106d1471d388fa3a96a094c3906e7860ed4ad5ec3d7acad9b9045`, temperatura 0,4409471 e limiar 0,445. Relatório em `resultados/diagnostico_interno_fase3_5_v1/`; teste OpenFake, calibração e benchmark externo não foram inferidos. A seleção já escolheu o modelo: estas métricas são descritivas, não independentes.
- A auditoria posterior dos erros, informada pelo usuário, conferiu as 719 imagens: **44/537** erros no Community Forensics–Small (13 FP, 31 FN) e **23/182** no OpenFake (13 FP, 10 FN). Metadados originais do Community Forensics não registram o formato; todos os arquivos processados foram lidos como JPEG. Há falsos negativos com `p_fake` perto de 0,02 e falsos positivos perto de 0,97; um pequeno ajuste de limiar não explica nem corrige esses casos. Os grupos sintéticos listados têm em geral 10–20 imagens, portanto suas taxas individuais são instáveis. Relatório em `resultados/auditoria_erros_internos_fase3_5_v1/`, conforme saída do Colab; nenhum teste externo foi acessado.
- O cruzamento classe × resolução da célula 14 mostrou que, na seleção do Community Forensics–Small, as **312 sintéticas** têm lado curto processado de 512–1.023 px, enquanto **207/225 reais** ficam abaixo de 512 px. Isso prova uma associação no dataset, não que o modelo a tenha usado. As 12 imagens visualizadas incluem fotos rotuladas reais classificadas como sintéticas e sintéticas fotográficas classificadas como reais; a inspeção visual não verifica a proveniência dos rótulos. A Fase 3.6 aumentou somente a frequência de `RandomDownUp` de 0,15 para 0,50, preservando checkpoint inicial da 3.3, dados, partições e demais parâmetros da 3.5. Código: `experimentos/fase3_6_escala.py`; notebook: `Notebooks/fase_3/rodar_fase3_6_colab.ipynb`; guia: `docs/README_fase3_6.md`. Segundo a saída do Colab fornecida pelo usuário, o treino da 3.6 concluiu 12 épocas com GPU Tesla T4 e partições idênticas às da 3.5: **3.594/719/719**. Melhor época **12**, AUC de seleção **0,9472355**, acurácia balanceada aparente de calibração **0,9075869**, temperatura **0,4662258** e limiar **0,495**. O teste OpenFake de 1.947 imagens foi apenas conferido por hash. Os oito artefatos esperados foram informados como presentes em `modelos/fase3_6_escala_p05_v1/`; o Drive não foi montado nesta sessão local.
- Fase atual: Fases 3.5 e 3.6 concluídas internamente e avaliadas no mesmo Mirage-Test externo, conforme saídas do usuário. A AUC de seleção da 3.6 ficou 0,00156 abaixo da 3.5; no Mirage, a acurácia caiu de 0,6045 para 0,572 e a AUC de 0,679972 para 0,6303885. Isso descreve a variante de escala, sem justificar ajuste da próxima fase pelo benchmark visto. A meta de 85% **não foi demonstrada**. A Fase 3.4 v2 atingiu 55% (11/20) no teste externo histórico de 20 imagens e 62,1% no RAISE/MNW de 2.000 imagens. Os modelos anteriores permanecem preservados.
- **Mirage-Test 2000 v1 avaliado**, conforme saída do Colab fornecida pelo usuário: 1.000 reais e 1.000 sintéticas, acurácia **0,6045** (1.209/2.000; IC Wilson 95% **0,58289–0,62571**), acurácia balanceada **0,6045**, AUC **0,679972**, precisão **0,7119675**, recall sintético **0,351**, especificidade real **0,858**, F1 **0,4701942**; TN=858, FP=142, FN=649, TP=351. Temperatura congelada **0,4409470856**, limiar **0,445**, checkpoint SHA-256 `1dd78d4cd32106d1471d388fa3a96a094c3906e7860ed4ad5ec3d7acad9b9045`, manifesto SHA-256 `4b3658f880e022957557933fd662c69485c19a5ed9a04e2e09c32ce485737190`, script SHA-256 `6b1f4ca329f1882fd401e6fdbbcada64cf74201a0ab55808b008c721e80b58a3`. Nenhuma imagem exigiu ampliação abaixo de 224 px. O resultado foi salvo em `resultados/benchmark_externo_mirage_2000_v1/fase3_5_cfsmall_v1/` no Drive, segundo a saída do usuário; esta sessão local não montou o Drive. A cópia local informou 2.000/2.000 hashes verificados. O conjunto agora é visto e não serve para escolher ou avaliar independentemente uma variante futura. Ausência de hash repetido não prova independência de origem após edições.
- Branch local: `fase3-ajuste-fino`, acompanhando `origin/fase3-ajuste-fino`. O snapshot registrado na execução do Colab foi `8fec4fff64cc064be5ae4c7892ba69137976ce70`.
- Código principal: `experimentos/fase3_4_finetune.py`. Interface de execução: `Notebooks/fase_3/rodar_fase3_4_colab.ipynb`. A arquitetura e funções da Fase 3.3 são reutilizadas de `Notebooks/fase_3/fase3.3_compressao.ipynb` no snapshot sincronizado do GitHub.
- O notebook histórico da Fase 1, `Notebooks/fase_1/fase1_vit.ipynb`, foi incorporado da cópia fornecida pelo autor. Somente a orientação inicial foi atualizada; o treino, as saídas salvas e o checkpoint existente não foram reexecutados ou alterados. Veja `Notebooks/fase_1/README.md`.

## Fase 3.4 v2: execução concluída

- Código novo: `experimentos/fase3_4_finetune_v2.py`, `experimentos/amostragem_fase3_4.py` e `experimentos/particoes_fase3_4.py`; interface: `Notebooks/fase_3/rodar_fase3_4_v2_colab.ipynb`; instruções: `docs/README_fase3_4_v2.md`.
- Estado: **smoke e treino concluídos no Colab conforme saídas fornecidas pelo usuário**. A v1 permanece concluída e inalterada. O checkpoint remoto não está disponível nesta sessão local.
- Inicialização: checkpoint da Fase 3.3 em `/content/drive/MyDrive/TCC_Final/modelos/fase3_3_compressao_v1/phase3_final.pt`. Novos artefatos: `modelos/fase3_4_finetune_v2_sqrt_sampler/`; smoke separado em `modelos/fase3_4_finetune_v2_sqrt_sampler_smoke/`.
- A mudança experimental é a amostragem com massa por grupo proporcional a `sqrt(n)` e classes 50/50. O manifesto da v1 é reutilizado; replay efêmero desativado para garantir retomada consistente. Como a seleção real de replay da v1 não foi persistida, a comparação causal entre v1 e v2 tem essa limitação.
- O `split_manifest.json` registra as partições efetivas. A retomada compara manifesto, partições, configuração e hashes de código; não reavalia nem modifica a v1. Smoke e treino da v2 já terminaram: não repetir as células 7 ou 8 para obter resultados.
- Smoke remoto informado: GPU Tesla T4, 1.668 imagens de treino, 182 de seleção, 182 de calibração, 1.947 de teste, replay zero, forward finito e `smoke.json` salvo. Levou 21,6 minutos, principalmente pela verificação de hashes no Drive. O log remoto do treino mostra a mensagem histórica `Fase 2 carregada`; o código carrega estritamente os pesos iniciais da Fase 3.3 depois de montar a arquitetura. A correção do texto foi publicada posteriormente no commit `37fc72c`, sem modificar os pesos já treinados.
- Treino remoto informado: épocas 1–9 concluídas em 3,0 minutos; early stopping após três épocas sem melhora; melhor checkpoint da época 6 por AUC de seleção 0,9671. A célula 9 confirmou `experiment_config.json`, `split_manifest.json`, `history.csv`, `last.pt`, `best.pt`, `phase3_final.pt`, `final_metrics.json`, `per_group.csv` e `test_predictions.csv` no diretório v2.
- Métricas OpenFake informadas: acurácia **0,7288136**, AUC **0,8222402**, acurácia balanceada **0,7258141**, precisão **0,7804552**, recall **0,6156283**, F1 **0,6883117**, especificidade **0,836**; TN=836, FP=164, FN=364, TP=583; temperatura **0,4402569**, limiar **0,37**. São resultados do teste OpenFake, não das 20 imagens externas.
- Avaliação externa congelada informada pelo usuário: **55% (11/20)** de acurácia, AUC **0,6875**, acurácia balanceada **62,5%**, precisão **100%** (somente três predições positivas), recall sintético **25%**, F1 **40%**, especificidade **100%**; TN=8, FP=0, FN=9, TP=3. Intervalo de Wilson de 95% para acurácia: **[34,21%, 74,18%]**. O Colab confirmou 20 arquivos distintos e nenhuma sobreposição exata com as 3.979 imagens OpenFake. O resultado está em `resultados/avaliacao_externa_fase3_4_v2_v1/` no Drive; esta sessão local não montou o Drive. Origem das imagens está marcada como `desconhecido` em `rotulos.csv`. Não recalibrar a v2 com esses casos.
- Em relação à v1, a v2 ganhou 1,34 ponto percentual de acurácia e 3,59 de recall, com 0,8 ponto a menos de especificidade. A comparação causal tem a limitação do replay da v1 não registrado.
- Verificações locais da v2: testes unitários sem GPU de amostragem e partições, sintaxe dos scripts e células conferida. Smoke e treino foram verificados pelas saídas do usuário, não pela montagem do Drive nesta sessão.
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

### Fase 3.5 (treino e diagnóstico interno concluídos)

1. Não repetir piloto, preparo, smoke ou treino da 3.5 apenas para obter novos números. `training_complete.json` indica conclusão; `auto` não continua épocas adicionais.
2. O diagnóstico da seleção registrou 26 falsos positivos e 41 falsos negativos. Conferir `per_group.csv` e `predictions.csv` antes de definir uma hipótese para a próxima variante. A calibração foi ajustada na própria partição B; sua acurácia balanceada é aparente. A origem real `n/a` limita conclusões sobre novas fontes reais.
3. Se houver hipótese sustentada pelo diagnóstico interno, criar experimento separado com uma única mudança. Congelar checkpoint, temperatura e limiar antes de qualquer nova avaliação externa. RAISE/MNW e as 20 imagens históricas já deram feedback e não servem para escolher a variante nem como teste final independente da 3.5.

### Fase 3.6 (treino interno concluído)

1. Não repetir a célula 8 do notebook da 3.6; `training_complete.json` foi informado como presente. O modo `auto` não acrescenta épocas a uma execução concluída.
2. Preservar o checkpoint da 3.6 em `modelos/fase3_6_escala_p05_v1/`. A comparação interna e a avaliação Mirage informada pelo usuário não indicam ganho sobre a 3.5. Não repetir a inferência para ajustar a próxima versão nem chamar o Mirage de teste cego independente.

### Benchmark externo RAISE/MNW (histórico concluído)

As sete avaliações foram executadas e salvas no Drive segundo saídas do usuário. Não repetir células de piloto, reparo, preparação ou avaliação para escolher a Fase 3.5. O benchmark não é mais um teste final independente para novos modelos que foram desenvolvidos após seu resultado.

### Experimentos Fase 3.4 já concluídos

1. Abra `Notebooks/fase_3/rodar_fase3_4_colab.ipynb` pelo VS Code e conecte a GPU do Colab. A configuração inicial da credencial já foi usada nas saídas salvas; não a refaça sem necessidade. Confirme que a branch remota contém a estrutura nova antes de sincronizar o snapshot.
2. Em uma sessão nova, execute as células 1 a 5 em ordem: dependências e GPU, montagem do Drive, senha de desbloqueio, sincronização do snapshot do GitHub e verificação dos checkpoints. Na sessão já conectada, execute somente as células cujas variáveis ou versões estejam faltando.
3. Confira no Drive `modelos/fase3_4_finetune_v1/final_metrics.json`, `history.csv`, `best.pt` e `last.pt`. A saída salva no notebook indica conclusão, mas esta cópia local não monta o Drive. Não execute novamente as células de preparação ou treino sem necessidade.
4. Para ler as métricas em nova sessão, rode as células 1–5 e depois a célula 9 de leitura de resultados. A célula 8 contém `executar_com_progresso('auto')` e é **manual**; se `final_metrics.json` existir, o modo `auto` apenas informa que o experimento está concluído.

## Problemas conhecidos e prioridades

- A 3.5 teve 60,45% de acurácia no Mirage-Test externo e a 3.6 teve 57,2%, conforme saídas do usuário; ambas abaixo da meta de 85%. As AUCs de seleção 0,9488 e 0,9472 e acurácias balanceadas aparentes de calibração 0,9213 e 0,9076 são internas; não compará-las diretamente com AUCs de outros conjuntos.
- O diagnóstico interno da 3.5 foi executado na GPU conforme saída do usuário, usando `experimentos/diagnostico_interno_fase3_5.py` e a célula 12 de `Notebooks/fase_3/rodar_fase3_5_colab.ipynb`. Fez inferência apenas na seleção, mantendo calibração e testes reservados sem nova inferência. O relatório permanece em `resultados/diagnostico_interno_fase3_5_v1/`.
- A auditoria de erros, a revisão visual da célula 14, o treino interno da 3.6 e as avaliações comparativas da 3.5 e 3.6 no Mirage-Test foram concluídos conforme saídas do usuário. Prioridade: usar as limitações identificadas **nos dados internos** (associação classe × resolução e sintéticas fotográficas difíceis) para definir uma única versão com dados reais e sintéticos mais variados. A origem `n/a` limita a auditoria, mas não impede inferência de imagens sem metadados. Cada fase futura terá comparação externa no Mirage congelado. Para afirmar desempenho em um teste cego independente de uma variante criada depois desse feedback, será preciso outro conjunto ainda não observado. Não usar as predições Mirage-Test, RAISE/MNW ou das 20 imagens históricas para decidir dados, pesos ou parâmetros. Preservar todos os checkpoints e resultados.
- A avaliação externa única da v2 foi executada no Colab; não repetir a célula 10 nem usar os 20 casos para recalibrar ou selecionar hiperparâmetros. O próximo experimento, se houver, deve ser separado e orientado pela auditoria de dados internos. Uma futura avaliação final independente exigirá outro conjunto externo reservado, pois estas 20 imagens já foram vistas como feedback de pesquisa.

- Uma falha anterior na preparação vinha de `drive.mount()` dentro do subprocesso. A correção está no commit `a54385b`; caso a falha reapareça, confirme que a célula 4 sincronizou esse commit ou um posterior.
- Após a reorganização, a célula 39 do notebook histórico da Fase 3.3 ainda consulta cópias antigas em `MyDrive/TCC_Final/Notebooks/fase3 - openfake/`. Ela é exploratória e não é executada pelo script da Fase 3.4; os caminhos do Drive foram preservados deliberadamente. A existência dessas cópias no Drive não foi verificada nesta sessão.
- O `.gitignore` bloqueia novos dados, modelos e credenciais, mas 62 arquivos de imagens/CSV externos já constam do índice Git. A reorganização não os removeu nem desindexou; um saneamento do histórico/publicação exige decisão separada.
- A validação local confirmou JSON de nove notebooks, sintaxe do script, índices de células e referências do Colab. Não há `torch`, `torchvision`, `datasets`, `sklearn` nem `cryptography` no Python local; importação funcional, GPU e sincronização remota dependem de teste no Colab com a estrutura nova.
- A saída anterior de `subprocess.run` escondia o erro interno até o fim da célula. O notebook local atual exibe logs durante a execução e avisos periódicos. Um aviso de processo ativo não prova avanço da coleta; acompanhe os logs e o manifesto.
- A diferença entre AUC de seleção interna da v2 (0,9671), AUC OpenFake (0,8222) e AUC externa (0,6875) reforça a preocupação com generalização, mas os conjuntos distintos não identificam a causa. A retomada após desconexão foi implementada por `last.pt`, mas não há registro de uma desconexão/retomada efetiva nesta execução.
- Antes de trocar de computador, confira que a branch local e `origin/fase3-ajuste-fino` apontam para o mesmo commit. Publicações futuras exigem autorização e não devem incluir credenciais, datasets, checkpoints nem resultados grandes no Git.
