# Auditoria interna da 3.8 e alternativas para a Fase 3.9

> **Atualização:** o piloto ampliado de novos pares AIGI teve apenas 5/60
> fotografias reais utilizáveis. O grupo decidiu verificar TGIF/TGIF2 como
> fonte de pares originais/alterados. A seleção A+C descrita abaixo é
> histórica; veja [`README_fase3_9_tgif.md`](README_fase3_9_tgif.md).

**Estado em 04/11/2026:** a Fase 3.8 concluiu treino e comparação Mirage.
Seu checkpoint final, o manifesto, a calibração e o resultado externo ficam
preservados nos diretórios originais do Drive. A auditoria por imagem descrita
aqui está **implementada, mas ainda não foi executada no Colab**. Não há
novas métricas internas por grupo até essa execução.

## O que já sabemos, e o que falta medir

- O treino 3.8 usou 6.152 imagens; seleção e calibração têm 1.420 cada.
  O teste OpenFake de 1.947 imagens foi conferido por hash, sem inferência.
- Nos dados processados de treino, 1.936 reais do Community Forensics tinham
  lado curto entre 256 e 511 px, enquanto 1.744 sintéticas tinham 512 px ou
  mais. Os 320 pares AIGI de treino têm JPEG e faixa de tamanho iguais nas
  duas classes. Isso mostra uma **associação de dados**, não prova que a rede
  usa resolução como atalho.
- A 3.8 teve AUC de seleção 0,9603094 e temperatura/limiar internos
  0,5229031/0,31. No Mirage já observado, teve 59,9% de acurácia e AUC
  0,641268; a 3.7 teve 62,3% e 0,6908315. Esses resultados são comparativos,
  não um teste cego para escolher as próximas amostras ou hiperparâmetros.
- Falta medir os **erros na seleção interna** por dataset, grupo, fonte real,
  resolução, formato e orientação. Essa partição escolheu o checkpoint;
  portanto, seus resultados também são descritivos e não uma estimativa
  independente de generalização.

## Preservação e execução da auditoria

O script [`auditoria_interna_fase3_8.py`](../experimentos/auditoria_interna_fase3_8.py)
confere os hashes da arquitetura, script de treino, manifestos e checkpoint,
e reconcilia as linhas das partições antes de abrir imagens. A célula de cópia
local verifica os hashes das imagens novas. O script usa temperatura e limiar já salvos,
inspeciona cabeçalhos de imagens apenas do treino e da seleção, e faz cinco
recortes **somente** na seleção interna. Não treina, não recalibra, não infere
calibração, teste OpenFake ou Mirage. O hash do checkpoint conhecido serve
apenas para conferir sua identidade; nenhuma predição externa é lida.

Após a publicação autorizada do código na branch `fase3-ajuste-fino`, abra
[`auditar_interno_fase3_8_colab.ipynb`](../Notebooks/fase_3/auditar_interno_fase3_8_colab.ipynb)
no VS Code com GPU do Colab:

1. Células **1–5**: ambiente, Drive, credencial GitHub, snapshot e progresso.
2. Célula **6**: copia 3.5, 3.7 e AIGI 3.8 do Drive para `/content` e verifica
   hashes. Em nova sessão, essa cópia local precisa ser refeita.
3. Célula **7**: executa manualmente a auditoria. Imprime progresso e grava
   exclusivamente em
   `/content/drive/MyDrive/TCC_Final/resultados/auditoria_interna_fase3_8_v1/`.
   Uma segunda execução lê o resumo existente; não o substitui.
4. Célula **8**: lê o resumo e os segmentos com 10 ou mais imagens. Envie a
   saída dessa célula e, se necessário, `segments_selection.csv`, sem enviar
   imagens ou credenciais.

Arquivos previstos: `summary.json`, `metadata_train_selection.csv`,
`distribution.csv`, `predictions_selection.csv`, `errors_selection.csv` e
`segments_selection.csv`. Checkpoints, históricos e resultados anteriores
permanecem em seus diretórios. Se uma execução falhar antes da gravação, o
diretório final não aparece; se falhar durante a gravação, uma pasta parcial
é preservada para inspeção e o script recusa sobrescrevê-la.

## Opções para a 3.9 — nenhuma escolhida ainda

| Opção | Mudança isolada | Possível benefício | Custo e limite | Critério interno para avançar |
| --- | --- | --- | --- | --- |
| **A. Diversidade real com Open Images V7** | Fazer o piloto já preparado, selecionar fotografias de autores e cenas variados e combiná-las com sintéticas de conteúdo comparável, sem reutilizar divisões reservadas. | Pode reduzir falsos positivos em fotografias diversas e enfraquecer associação entre classe, fonte e tamanho. | Download seletivo e conferência de licença/autoria por imagem; rótulo de foto real exige revisão, pois metadados não provam ausência de edição por IA. | Auditoria interna mostrar lacuna em fontes/cenas reais e piloto confirmar imagens legíveis, espaço, licença e isolamento por autor. |
| **B. Novos shards do Community Forensics–Small** | Selecionar pares adicionais de shards ainda não usados, com quantidades e faixas de resolução por classe fixadas antes do treino. | Amplia geradores e cenas com menor trabalho de obtenção de reais, podendo reduzir concentração em poucos grupos e o atalho de resolução. | Shards Parquet precisam ser baixados inteiros; o acervo completo é grande e a licença é não comercial. `real_source=n/a` dos shards anteriores não pode ser reinterpretado como fonte conhecida. | Auditoria apontar grupos ou faixas pouco representados; piloto de dois shards confirmar contagens, diversidade, espaço e ausência de duplicatas. |
| **C. Mais pares AIGI com UIDs novos** | Ampliar pares reais/sintéticos de geradores e cenas já documentados, mantendo o mesmo UID em uma só divisão e processamento equivalente nas duas classes. | Dá exemplos de conteúdo semelhante com rótulos opostos, útil para testar se a rede aprende sinais da geração em vez do tema da cena. | Fotos reais são buscadas nas fontes e podem faltar; mais pares da mesma família não garantem diversidade. Os termos das fontes reais precisam de conferência. | Auditoria interna mostrar benefício por grupo AIGI e erros concentrados em grupos com poucos pares; piloto confirmar URLs, hashes e quota antes de congelar. |
| **D. Controle de escala com os dados existentes** | Criar versão isolada com amostragem ou transformações que aproximem a distribuição de resolução entre classes, mantendo fontes e divisão fixas. | Custo de download baixo; testa diretamente a hipótese de atalho de tamanho. | Pode perder artefatos finos e já houve aumento de `RandomDownUp` na 3.6 sem melhora interna. | Auditoria interna evidenciar associação forte e erros ligados à faixa de tamanho; definir uma única transformação antes do treino. |

As opções são **hipóteses**, sem promessa de atingir 85%. Após a auditoria
interna da 3.8, o grupo escolheu combinar **A + C** como hipótese de dados da
3.9. Os pilotos estão em `Notebooks/fase_3/pilotos_fase3_9_colab.ipynb` e o
procedimento em `docs/README_fase3_9.md`. Quantidades e divisões ainda
dependem do piloto; não usar predições Mirage para defini-las.
A [página oficial do Open Images V7](https://storage.googleapis.com/openimages/web/download_v7.html)
oferece metadados e download por seleção; o
[cartão do Community Forensics–Small](https://huggingface.co/datasets/OwensLab/CommunityForensics-Small)
descreve seu formato e licença; o
[cartão do AIGI](https://huggingface.co/datasets/Coxy7/AIGI-Detection-Quality-Paradox)
explica os UIDs pareados e que as fotos reais não vêm no repositório.

Em qualquer opção: piloto verifica viabilidade, preparo fixa divisões e
hashes, smoke verifica carregamento e forward, treino ajusta pesos, seleção
escolhe época, calibração interna fixa temperatura/limiar e a comparação
Mirage ocorre uma vez após congelamento. A retomada usa `last.pt` só se o
treino estiver incompleto. Para demonstrar 85% **independentemente**, será
necessário um conjunto externo novo, reservado antes de ser observado.
