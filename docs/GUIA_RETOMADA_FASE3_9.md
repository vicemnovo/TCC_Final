# Retomada do TCC: Fase 3.9 e próximo experimento

Atualizado em 04/11/2026. Este guia usa as saídas salvas no notebook Colab e
os arquivos do repositório. O Drive remoto não foi montado nesta revisão.

## Ponto exato da pausa

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

**Ainda não há avaliação externa da 3.9.** A tentativa de executar a cópia
Mirage parou em `NameError: CODE_ROOT` antes de copiar ou inferir imagens,
porque as células de preparação não haviam sido executadas no novo kernel.
Não há acurácia Mirage 3.9 registrada nesta pausa. A AUC de seleção acima
não é uma estimativa de acurácia externa; a seleção da 3.9 também difere da
3.8 e seus valores não devem ser comparados como se fossem o mesmo teste.

## Primeira ação ao voltar: concluir a comparação Mirage

Abra `Notebooks/fase_3/rodar_fase3_9_tgif_colab.ipynb` no VS Code e conecte
o runtime Colab com GPU. Se for **um runtime novo**, execute as células de
código **#1 a #5 em ordem**: dependências/GPU, montagem do Drive,
desbloqueio da credencial GitHub, snapshot da branch e definição de
`run_live`. Confirme as mensagens `Código sincronizado` e `Código:`. A
credencial é solicitada novamente em cada sessão; não fica no notebook.

Depois execute somente:

1. **#11 — cópia Mirage:** traz as 2.000 imagens congeladas do Drive para
   `/content` e confere hashes. É a cópia local para inferência, não treino.
2. **#12 — congelamento:** registra o SHA do checkpoint 3.9, temperatura,
   limiar, código e manifesto **antes** de abrir as imagens para inferência.
3. **#13 — avaliação externa comparativa:** calcula acurácia, AUC, recall
   sintético, especificidade real, matriz de confusão e comparação pareada
   com a 3.8. Salva em um diretório novo no Drive.
4. **#14 — leitura:** mostra o resultado salvo, sem recalibrar.

As células acima aparecem após um título Markdown no notebook; procure os
comentários `# 11` a `# 14` no início do código. Se a avaliação já existir
quando voltar, leia **#14**; o script não deve substituir o resultado.
Se a sessão cair antes de terminar, reconecte, execute **#1 a #5** e retome
pela primeira dessas quatro etapas ainda pendente. Não repita preparo
TGIF, smoke ou treino (`#6` a `#10`); o treino está marcado como completo.

O Mirage tem 1.000 fotos reais e 1.000 sintéticas fixas. Ele já foi visto
nas fases 3.5–3.8: sua avaliação da 3.9 será **comparativa**, não um teste
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
O próximo experimento só deve ser fixado depois da auditoria interna e do
piloto da fonte escolhida. A avaliação Mirage deve ser registrada para
comparação, sem orientar mudanças posteriores.

## Conferência do Git ao voltar

No projeto principal, execute `git status -sb` e `git log -1 --oneline`.
Confira que a branch é `fase3-ajuste-fino` e que o commit local acompanha
`origin/fase3-ajuste-fino`. No repositório da demo, a branch é `main`.
Datasets, checkpoints, credenciais e resultados grandes ficam no Drive,
fora de ambos os repositórios Git. Preserve os diretórios das fases antigas.
