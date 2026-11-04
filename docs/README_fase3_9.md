# Fase 3.9 — piloto combinado Open Images + AIGI

> **Histórico:** a prova ampliada AIGI terminou com 0/20, 0/20 e 5/20 fotos
> reais utilizáveis nos três grupos. O grupo decidiu verificar TGIF/TGIF2
> como fonte de novos pares. O procedimento atual está em
> [`README_fase3_9_tgif.md`](README_fase3_9_tgif.md); não repetir os pilotos
> AIGI abaixo para decidir esta versão.

## Hipótese e estado

A auditoria interna da 3.8, informada pelo grupo, registrou 74 falsos
positivos e 43 falsos negativos em 1.420 imagens da seleção. As fotos reais
de `pexels` (15/52), `aigi:PA_alpha` (22/80) e OpenFake com lado curto de 512 px
ou mais (22/80) tiveram erros relevantes. Esses grupos podem se sobrepor
nas tabelas por dimensão; não somar suas contagens. A seleção já participou
da escolha do checkpoint: essas associações são descritivas e não provam
qual atalho a rede usa. O Mirage observado não foi usado para definir a 3.9.

A hipótese combinada é ampliar a variedade de fotografias reais, com
Open Images V7, e acrescentar pares reais/sintéticos AIGI **novos**, com
UIDs diferentes dos 480 pares da 3.8. A mudança planejada é somente de
dados. Arquitetura, checkpoint inicial, protocolo de recortes e demais
parâmetros serão comparados com os scripts existentes antes de criar o
treinador. **Ainda não há seleção congelada, treino, smoke ou resultado 3.9.**

## Primeiro piloto concluído; próximo passo sem treino

Saída informada pelo grupo: Open Images teve **1.739.728 imagens elegíveis nos
metadados**, **76.961 autores declarados** e 10/10 JPEGs legíveis no piloto.
Isso confirma acesso e quantidade de candidatos, não a qualidade do rótulo
das imagens ainda não examinadas. No AIGI, havia 129 UIDs novos pareáveis em
`FLUX.1_dev`, 129 em `PA_alpha` e 311 em `SD_2.1`; de cinco fotografias reais
testadas em cada grupo, ficaram utilizáveis **0, 0 e 2**, respectivamente.
As falhas incluíram URLs inacessíveis, imagens ilegíveis e lado curto abaixo
de 256 px. Quinze provas são insuficientes para fixar o tamanho ou descartar
o AIGI. Os relatórios existentes permanecem preservados.

**Execute agora** `Notebooks/fase_3/piloto_ampliado_fase3_9_colab.ipynb`:
células 1–5 para montar o ambiente e sincronizar o código; célula 6 para
verificar **20** fotos reais por grupo nos três grupos anteriores, mantendo
o relatório original; célula 7 para mostrar o resumo. GPU não é necessária.
O novo arquivo local é
`/content/fase3_9_aigi_piloto/pilot_expanded_20_summary.json`.
Envie a saída da célula 7 e eventual erro da 6. Se a sessão anterior do
Colab ainda existir, o cache de CSV, shards e fotos já baixadas será
reutilizado; em sessão nova, os arquivos completos serão baixados novamente.

## Procedimento do primeiro piloto (histórico)

Abra `Notebooks/fase_3/pilotos_fase3_9_colab.ipynb` no runtime do Colab.
GPU não é necessária neste passo. Execute as células em ordem:

1. **1–5:** dependências, Drive, credencial GitHub, snapshot da branch e
   função que imprime progresso. O código precisa estar publicado na branch
   `fase3-ajuste-fino`; o Colab não lê o disco `E:`.
2. **6:** piloto Open Images. Baixa o CSV oficial completo e dez JPEGs para
   `/content`, de autores declarados distintos. Pode demorar no primeiro
   download. Metadados e inspeção técnica não comprovam, por si, que cada
   imagem não foi manipulada.
3. **7:** piloto AIGI. Lê o manifesto congelado da 3.8, baixa o CSV e shards
   completos do commit AIGI fixado e conta UIDs novos nos grupos
   `FLUX.1_dev`, `PA_alpha` e `SD_2.1`. Tenta cinco fotografias reais por
   grupo. Se a credencial HF cifrada já existir no Drive, pede sua senha;
   nenhuma senha ou token aparece no notebook ou saída. Caso contrário,
   tenta acesso público. Nenhum par é selecionado para treino.
4. **8:** mostra apenas contagens e falhas relevantes. Envie essa saída e
   qualquer erro das células 6–7. Os relatórios locais ficam em
   `/content/fase3_7_openimages_pilot/pilot_summary.json` e
   `/content/fase3_9_aigi_piloto/pilot_summary.json`; `/content` é
   temporário. Não envie credenciais ou URLs privadas.

Os pilotos baixam arquivos completos antes de examinar exemplos, conforme
`docs/POLITICA_DADOS_FUTUROS.md`. Não usam `streaming=True`, imagens Mirage,
checkpoints ou avaliações externas. Não criam divisões, dados persistentes
de treino ou modelos.

## Critérios para continuar

- Open Images: JPEGs legíveis, autores suficientes para separar divisões,
  licença documentada e diversidade real de cenas. A seleção final precisará
  de revisão visual/amostral de rótulos, evitando tratar a fonte declarada
  como prova automática de fotografia não alterada.
- AIGI: candidatos novos por grupo, fotografias reais efetivamente baixáveis,
  pares completos e UIDs ausentes da 3.8. A disponibilidade precisa ser
  estimada antes de fixar uma quantidade; `PA_alpha` não será ampliado
  automaticamente só por constar na seleção anterior.
- Conjunto combinado: definir números e divisão **antes** do treino, manter
  pares AIGI juntos, separar autores reais e geradores quando possível,
  verificar hashes/duplicatas com todas as divisões anteriores e equilibrar
  classe, fonte, formato e resolução. Fotos Open Images sozinhas não devem
  criar uma pista trivial de classe ou origem.
- Se uma fonte falhar, interromper o plano e registrar o motivo. Não trocar
  dados em resposta às predições do Mirage.

Depois dos pilotos, o **preparo/congelamento** persistirá arquivos escolhidos,
manifesto e hashes no Drive. O **smoke test** verificará GPU, arquitetura,
checkpoint inicial, dados, forward e salvamento sem treinar. O **treino**
ajustará pesos em diretório 3.9 separado e poderá retomar de `last.pt` se
interrompido. A **seleção interna** escolherá a época; a **calibração interna**
fixará temperatura e limiar. Só então haverá **avaliação externa comparativa**
congelada no Mirage. O Mirage já foi observado, portanto não demonstrará
sozinho a meta independente de 85%.

## Fontes

- [Open Images V7: download de subconjuntos](https://storage.googleapis.com/openimages/web/download_v7.html)
- [AIGI-Detection-Quality-Paradox: cartão do dataset](https://huggingface.co/datasets/Coxy7/AIGI-Detection-Quality-Paradox)
