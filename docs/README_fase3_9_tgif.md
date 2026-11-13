# Fase 3.9 — avaliação de viabilidade TGIF/TGIF2

> Registro histórico do piloto, já concluído conforme saída do Colab:
> 341 IDs pareados no TGIF `validation`, sem treino iniciado. Para continuar,
> use o [roteiro atual da 3.9](README_fase3_9_execucao.md). As instruções
> de execução do piloto abaixo permanecem apenas para documentação.

## Decisão técnica

O grupo pediu uma fonte de pares que dispense o AIGI. O TGIF/TGIF2 dos
autores do IDLabMedia é **pertinente** ao detector: fornece fotografias
originais do MS-COCO e versões alteradas por inpainting/geração, com um ID
COCO no nome do original e de suas alterações. O TGIF inclui SD2, SDXL e
Adobe; o TGIF2 acrescenta variantes FLUX. O conjunto tem versões de
alteração localizada (`sp`) e regeneração completa (`fr`). A licença
declarada pelos autores é CC BY-SA 4.0. Essa fonte adiciona uma classe de
manipulação que o piloto AIGI não conseguiu ampliar de modo equilibrado.

**Isto é validade de fonte e protocolo, não resultado de modelo.** A próxima
ação é um piloto de acesso e pareamento. O AIGI novo ficou com 0/20 fotos
reais utilizáveis em FLUX.1_dev, 0/20 em PA_alpha e 5/20 em SD_2.1; os
480 pares AIGI históricos continuam preservados. Não usar o Mirage para
selecionar exemplos TGIF ou parâmetros.

## Acesso e custo verificados sem baixar o acervo

As páginas oficiais apontam para três compartilhamentos públicos Nextcloud.
Uma consulta WebDAV somente leitura confirmou arquivos separados por
`orig`, `sd2-fr`, `sd2-sp`, `sdxl-fr` e grupos FLUX, cada um com `training`,
`validation` e `testing`. O servidor respondeu a HEAD e a um pedido HTTP
Range de 1 KiB; **não foram baixados os arquivos grandes nesta verificação**.

| Par de arquivos | Tamanho comprimido aproximado | Uso planejado |
| --- | ---: | --- |
| TGIF `orig_validation` + `sd2-fr_validation` | 1,76 GB | Piloto de acesso e estrutura, sem treino |
| TGIF `orig_training` + `sd2-fr_training` | 11,56 GB | Fonte possível para o preparo interno |
| TGIF2 `orig-flux_training` + `flux1schnell-fr_training` | 19,86 GB | Expansão possível, sujeita a quota e piloto |

Esses tamanhos são dos **arquivos completos** no servidor, não do subconjunto
final no Drive. A política do projeto exige baixar os arquivos escolhidos
por inteiro para `/content` antes de ler exemplos. Se não couberem, reduzir
o escopo **antes** de preparar ou treinar; não substituir por streaming.
As cópias persistidas no Drive serão apenas das imagens selecionadas, com
manifesto, hashes e divisões novas. Não usar `testing` do TGIF/TGIF2 como
fonte de treino. A divisão `validation` dos autores, usada no piloto, não
entrará no treino da 3.9.

## Execute agora: piloto TGIF, sem GPU

Abra `Notebooks/fase_3/fase3_9/piloto_fase3_9_tgif_colab.ipynb` no Colab:

1. **Células 1–5:** dependências, Drive, desbloqueio GitHub, sincronização
   do commit publicado e função de progresso.
2. **Célula 6:** consulta o tamanho dos arquivos de treino TGIF/TGIF2,
   baixa integralmente os dois arquivos `validation` do TGIF para `/content`,
   retoma apenas `.part` se houver desconexão, confere SHA-256, lê os tar.gz
   sem extrair e verifica IDs e amostras de imagens. Exige cerca de 2,76 GB
   livres, incluindo margem de 1 GB. Pode demorar e mostra progresso.
3. **Célula 7:** mostra `pilot_summary.json`. Envie a saída e quaisquer erros
   da célula 6. O relatório fica em `/content/fase3_9_tgif_piloto/`; `/content`
   é temporário. Se a sessão cair antes do fim, execute novamente a célula
   6 após refazer as células de ambiente; ela retoma o arquivo parcial.

O **piloto** verifica a fonte, não mede acurácia. Depois dele, o
**preparo/congelamento** selecionará somente arquivos `training`, fixará
originais e alterações por ID na mesma divisão, verificará sobreposição
exata com OpenFake, Community Forensics, AIGI e benchmarks reservados, e
equilibrará formato e resolução. O **smoke test** verificará arquitetura,
checkpoint, forward, dados e gravação sem treinar. O **treino** ajustará
pesos em novo diretório e poderá retomar de `last.pt`. A **seleção** escolherá
a época; a **calibração interna** fixará temperatura e limiar. Só então
ocorrerá avaliação comparativa no Mirage congelado. O Mirage já é observado
e não comprova sozinho a meta de 85% em teste independente.

Os checkpoints, manifestos e resultados das fases anteriores não serão
sobrescritos. O código de preparo e treino da 3.9 só deve ser finalizado
depois de confirmar o formato real dos arquivos no piloto.

## Fontes primárias

- [TGIF/TGIF2, repositório dos autores](https://github.com/IDLabMedia/tgif-dataset)
- [Nextcloud: WebDAV de compartilhamentos públicos](https://docs.nextcloud.com/server/stable/developer_manual/client_apis/WebDAV/basic.html)
