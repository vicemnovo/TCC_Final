# Retomada do TCC: Fase 3.9 e próximo experimento

Atualizado em 10/11/2026. Este guia usa as saídas salvas no notebook Colab e
os arquivos do repositório. O Drive remoto não é acessível nesta revisão local.

## Ponto exato da pausa

**Antes de iniciar a 3.9.1**, corrigir V2 e revisar os candidatos visuais
internos no [notebook final da 3.9](../Notebooks/fase_3/fase3_9_final_colab.ipynb).
V2, V3 e fechamento já foram executados, mas o fechamento marcou
`revisao_tecnica_pendente`. A V2 tinha um erro de leitura do manifesto TGIF,
agora corrigido; a V3 não precisa ser repetida. Após publicar o código,
execute no Colab **1–5, 7, 9, 10, 12 e 13**. A célula 12 reutiliza o V3
gravado; a 13 cria um `.log` datado no Drive com as saídas e os relatórios.
Veja [`README_fase3_9_fechamento.md`](README_fase3_9_fechamento.md). Não
repetir preparo, treino ou avaliação Mirage.

O código principal está na branch `fase3-ajuste-fino` do repositório privado
`vicemnovo/TCC_Final`. A Fase 3.9 acrescentou **768 pares TGIF original +
SD2 totalmente regenerado**, com seleção por ID COCO e caminho de categoria.
Os pares foram divididos em 512 para treino, 128 para seleção de época e 128
para calibração. O experimento preserva a arquitetura ViT-Tiny + FFT +
EfficientNet-B0, parte do checkpoint da Fase 3.3 e grava novos artefatos em
`/content/drive/MyDrive/TCC_Final/modelos/fase3_9_tgif_sd2fr_v1/`.

O **piloto** verificou acesso e pareamento da fonte, sem treinar. O
**preparo/congelamento** fixou os pares e seus hashes no Drive. O **smoke test**
verificou GPU, divisões, checkpoint, forward e gravação; não mediu acurácia.
O **treino interno** executou 12 épocas e escolheu a época 10 pela seleção
interna (AUC 0,8988236). A **calibração interna** gravou temperatura 0,9187889
e limiar 0,47; sua acurácia balanceada aparente foi 0,8557320. O notebook
confirmou `phase3_final.pt`, `best.pt`, `last.pt`, `internal_metrics.json`,
`history.csv`, `split_manifest.json` e `training_complete.json`. O treino
usou 7.176 imagens, a seleção 1.676 e a calibração 1.676. As 1.947 imagens
do teste OpenFake foram apenas conferidas por hash, sem inferência.

**Avaliação Mirage concluída:** o grupo executou as células #11 a #14 após
configurar um novo runtime. O checkpoint, a temperatura e o limiar foram
congelados antes da inferência. Nas 2.000 imagens, a 3.9 obteve acurácia
**0,5875**, AUC **0,6328105**, recall sintético **0,431** e especificidade
real **0,744**; TN=744, FP=256, FN=569 e TP=431. A comparação pareada com a
3.8 registrou 1.063 acertos conjuntos, 135 exclusivos da 3.8, 112 da 3.9
e 690 erros conjuntos. O resultado está em
`/content/drive/MyDrive/TCC_Final/resultados/benchmark_externo_mirage_2000_v1/fase3_9_tgif_sd2fr_v1/`.
A AUC de seleção acima não é estimativa de acurácia externa; a seleção da
3.9 também difere da 3.8. O Mirage já foi observado e não comprova a meta
em um teste cego independente.

## Primeira ação ao voltar: definir a Fase 3.9.1

Não repita preparo, smoke, treino ou avaliação da 3.9 (células #6 a #14 do
notebook de execução), nem a auditoria interna já concluída. Preserve
`phase3_final.pt`, o lock da avaliação, `metrics.json` e o relatório de
auditoria. O [notebook de auditoria](../Notebooks/fase_3/auditar_interno_fase3_9_colab.ipynb)
teve suas células #7–#11 executadas pelo grupo; a #10 mostra apenas erros
visuais extremos e a #11 conta todos os erros. O
[relatório completo](AUDITORIA_INTERNA_FASE3_9.md) registra os resultados.

Na seleção interna da 3.9 houve **59 falsos positivos (FP) e 182 falsos
negativos (FN)** em 1.676 imagens: 1.435 acertos (**85,62%**). As sintéticas
TGIF contribuíram com 115 FN em 128 imagens; os FP se distribuíram entre
Community Forensics (18), OpenFake (16), TGIF (14) e AIGI (11). Essa seleção
escolheu o checkpoint e não comprova desempenho externo. O Mirage comparativo
já observado ficou em 58,75%; não usar seus erros para definir a 3.9.1.

Ao retomar em outro computador ou sessão:

1. Abra o repositório na branch `fase3-ajuste-fino` e execute
   `git status -sb` e `git pull --ff-only origin fase3-ajuste-fino` se a cópia
   local estiver limpa. Leia este guia, `ESTADO_PROJETO.md` e a auditoria.
   Para espelhar os `.ipynb` e `.py` no Drive, abra
   `Notebooks/configuracao/sincronizar_codigo_drive_colab.ipynb` e execute
   as células 1–5; confira `codigo_sync/<commit>.json`. Essa operação não
   precisa de GPU nem repete treinamento. Veja
   `docs/SINCRONIZACAO_CODIGO_DRIVE.md`.
2. Defina **uma hipótese controlada** para a 3.9.1. A primeira candidata é
   corrigir a baixa massa de amostragem das sintéticas TGIF (1,740% contra
   7,835% para TGIF real), mantendo dados, divisões, arquitetura, checkpoint
   inicial e demais parâmetros. Essa desigualdade foi medida, mas sua causa
   nos erros ainda precisa de experimento. Não há notebook 3.9.1 criado.
3. Implemente a variante em arquivos e diretório novos. Antes de usar GPU,
   teste a massa esperada do sampler, os rótulos e a separação de pares. O
   **smoke test** verificará dados, arquitetura, checkpoint, forward e gravação;
   ele não mede acurácia nem inicia o treino completo.
4. Inicie o **treino** manualmente apenas após o smoke. Selecione o checkpoint
   pela seleção interna e ajuste temperatura/limiar apenas na calibração
   interna. Compare nas mesmas imagens internas acurácia total, FP, FN,
   recall sintético, especificidade real, acurácia balanceada e AUC; não
   aceite ganho de FN que custe mais FP e reduza o total de acertos.
5. Com checkpoint, temperatura e limiar congelados, faça a avaliação Mirage
   comparativa em resultado novo, preservando os anteriores. Como o Mirage
   já foi visto, uma alegação final de 85% exigirá outro conjunto externo
   independente definido antes de observar suas predições.

Em avaliações **por imagem**, informe probabilidades estimadas de real e
sintética, classe pelo limiar interno congelado e frase pronta usando
`experimentos/linguagem_predicao.py`. As faixas da frase não mudam a decisão
nem representam certeza de origem. Resultados históricos não são regravados.

## Como a avaliação Mirage foi executada

Abra `Notebooks/fase_3/rodar_fase3_9_tgif_colab.ipynb` no VS Code e conecte
o runtime Colab com GPU. Se for **um runtime novo**, execute as células de
código **#1 a #5 em ordem**: dependências/GPU, montagem do Drive,
desbloqueio da credencial GitHub, snapshot da branch e definição de
`run_live`. Confirme as mensagens `Código sincronizado` e `Código:`. A
credencial é solicitada novamente em cada sessão; não fica no notebook.

Na sessão já concluída, o grupo executou:

1. **#11 — cópia Mirage:** traz as 2.000 imagens congeladas do Drive para
   `/content` e confere hashes. É a cópia local para inferência, não treino.
2. **#12 — congelamento:** registra o SHA do checkpoint 3.9, temperatura,
   limiar, código e manifesto **antes** de abrir as imagens para inferência.
3. **#13 — avaliação externa comparativa:** calcula acurácia, AUC, recall
   sintético, especificidade real, matriz de confusão e comparação pareada
   com a 3.8. Salva em um diretório novo no Drive.
4. **#14 — leitura:** mostra o resultado salvo, sem recalibrar.

As células acima aparecem após um título Markdown no notebook; procure os
comentários `# 11` a `# 14` no início do código. Para consultar o resultado
em outro runtime, execute **#1 a #5** e depois apenas **#14**.

O Mirage tem 1.000 fotos reais e 1.000 sintéticas fixas. Ele já foi visto
nas fases 3.5–3.8: sua avaliação da 3.9 é **comparativa**, não um teste
cego independente. Não usar suas predições para treinar, calibrar, escolher
pesos, dados ou hiperparâmetros. A meta de 85% em conjunto externo
independente ainda não foi demonstrada.

## Demonstração para apresentação

O repositório **privado e independente** `vicemnovo/TCC_Detector_Imagens_Demo`
contém `demo_fase3_9.ipynb`. Localmente está em
`E:\TCC\TCC_Detector_Imagens_Demo`. Abra o notebook com a mesma conta do
Drive, execute as células **1–5** e confira `Estado: final_calibrado`.
O notebook escolhe automaticamente `phase3_final.pt` quando encontra
`training_complete.json`; copia os pesos para `/content`, exige carregamento
estrito da arquitetura e usa cinco recortes por imagem. Exibe estimativas
para real/sintética e aplica o limiar 0,47 gravado internamente. Caso só
existisse `best.pt`, exibiria apenas um softmax ainda sem calibração.

Para imagens escolhidas no navegador Colab, execute a célula **6**. No
runtime iniciado pelo VS Code, execute a **7**: ela cria a pasta
`/content/drive/MyDrive/TCC_Final/apresentacao_fase3_9/entrada`; coloque
JPG, PNG ou WebP nela pelo Drive e execute a célula 7 novamente. O notebook
faz apenas inferência. A exibição de uma porcentagem não prova autoria ou
veracidade da cena. A inferência da demo ainda precisa de um teste prático
no Colab; a validação local foi estática, sem PyTorch instalado nesta máquina.

## Possibilidades para a próxima fase

Definir **uma hipótese por versão**, com piloto, preparo, smoke, treino,
seleção, calibração e comparação externa separados. Decidir os dados e os
parâmetros apenas por auditoria **interna** e viabilidade de fonte, não pelos
erros do Mirage. Três caminhos possíveis:

1. **TGIF2/FLUX pareado:** amplia geradores e cenas editadas mantendo pares
   real/sintético. Exige piloto de conteúdo e espaço antes de baixar os
   arquivos de treino (os dois candidatos consultados somam cerca de
   19,86 GB comprimidos). Benefício potencial alto; custo de coleta alto.
2. **Edições localizadas TGIF (`sp`) ou SDXL:** complementa a regeneração
   total SD2 da 3.9 com manipulações parciais. Reaproveita a lógica de
   pareamento por ID/categoria, mas requer auditoria de rótulos, máscaras,
   formato e resolução. Custo moderado a alto, dependendo dos arquivos.
3. **Fotografias reais de fonte visualmente diversa:** um piloto de Open
   Images V7 já leu dez JPEGs, mas isso não valida rótulos ou independência
   para treino. Uma seleção versionada com licença, controle de autores,
   duplicatas e formatos pode reduzir a dependência das fontes reais atuais.
   Custo de preparo moderado; benefício depende da qualidade da seleção.

Em qualquer opção, auditar o atalho de resolução observado nos dados
Community Forensics históricos: muitas reais têm lado curto 256–511 px,
enquanto muitas sintéticas têm 512 px ou mais. Os pares TGIF novos foram
normalizados em ambas as classes, mas não eliminam o viés dos dados antigos.
Para uma nova fonte, faça piloto antes do preparo e do treino. Para a hipótese
do sampler, teste primeiro a mudança isolada com os dados internos já
congelados. A avaliação Mirage deve ser registrada para comparação, sem
orientar mudanças posteriores.

## Conferência do Git ao voltar

No projeto principal, execute `git status -sb` e `git log -1 --oneline`.
Confira que a branch é `fase3-ajuste-fino` e que o commit local acompanha
`origin/fase3-ajuste-fino`. No repositório da demo, a branch é `main`.
Datasets, checkpoints, credenciais e resultados grandes ficam no Drive,
fora de ambos os repositórios Git. Preserve os diretórios das fases antigas.
