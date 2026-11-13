# Fase 3.5 — novos dados internos, sem reutilizar o benchmark externo

## Por que esta versão existe

No benchmark externo congelado RAISE/MNW (1.000 reais + 1.000 sintéticas), a
Fase 3.4 v2 acertou **62,1%** e teve recall sintético de **28,9%**. A precisão
de **86,0%** não representa a meta de 85% de *acurácia*: o detector classificou
711 sintéticas como reais. O novo benchmark já foi observado e **não pode ser
usado para escolher dados, checkpoint, arquitetura, limiar ou hiperparâmetros**
da 3.5, nem chamado de teste final independente da 3.5.

A 3.5 mantém a arquitetura ViT-Tiny + FFT + EfficientNet-B0, o treinador da
Fase 3.3, a política de amostragem da 3.4 v2 e os mesmos aumentos moderados.
O fator experimental principal é **adicionar dados internos com geradores
identificados**; a fonte individual das fotos reais veio como `n/a`. O checkpoint inicial continua o da Fase 3.3,
fixado antes da avaliação da 3.5. Não retreina nem altera as Fases 1–3.4.

## Fonte escolhida para o piloto

[Community Forensics–Small](https://huggingface.co/datasets/OwensLab/CommunityForensics-Small)
é publicada pelo grupo responsável pela pesquisa, contém rótulo, nome do
gerador, campo de fonte real e bytes da imagem. **No shard real testado, o campo
individual `real_source` veio como `n/a`; não tratá-lo como fonte conhecida.**
A licença informada é **CC-BY-NC-SA 4.0,
para pesquisa não comercial**. Seu uso no TCC experimental não concede
automaticamente permissão para uma API comercial. O conjunto inteiro ocupa
cerca de 260 GB e **não cabe** na quota gratuita; por isso, o piloto baixa
somente arquivos Parquet explicitamente escolhidos, por inteiro, em `/content`.
Não usa `streaming=True`, não baixa o acervo completo e não instala serviço pago.
O piloto remoto dos dois shards iniciais informou 2.939 sintéticas em
`HFCF_small_0.parquet` e 2.992 reais em `HFCF_small_117.parquet`; 77 grupos
sintéticos têm pelo menos 20 imagens, mas as reais têm origem individual
indisponível. Nenhuma seleção foi congelada ou treinada nesse piloto.

O artigo [Community Forensics](https://arxiv.org/abs/2411.04125) fundamenta
o foco em diversidade de geradores. O estudo
[Fake or JPEG?](https://arxiv.org/abs/2403.17608) alerta para atalhos de
compressão e resolução. Por isso, os dados novos de ambas as classes são
normalizados para JPEG RGB qualidade 90, lado maior limitado a 1024 px quando
possível e lado menor ampliado para pelo menos 224 px; a origem e o hash dos
bytes originais permanecem no manifesto. Isso reduz um atalho simples, mas
não comprova ausência de viés sem auditoria adicional.

## Arquivos e isolamento

- `Notebooks/fase_3/fase3_5/fase3_5_dados.py`: `pilot`, `prepare` e `stage`.
- `Notebooks/fase_3/fase3_5/fase3_5_finetune.py`: `smoke`, `train`, `resume` e `auto`.
- `Notebooks/fase_3/fase3_5/rodar_fase3_5_colab.ipynb`: interface Colab.
- Dados persistentes: `dados_fase3_5/fase3_5_novos_v1/` no Drive; enquanto
  incompleto, `.incompleto/`. Cópia efêmera: `/content/fase3_5_novos_v1/`.
- Novo experimento: `modelos/fase3_5_cfsmall_v1/`; smoke separado em
  `modelos/fase3_5_cfsmall_v1_smoke/`.

`prepare` fixa o commit do dataset, shards, seleção, divisões por grupo e hashes
em `plan.json`/`manifest.json`. Rejeita duplicatas exatas e sobreposição exata
com OpenFake, benchmark RAISE/MNW e imagens históricas localizadas no Drive.
Não examina predições externas. As sintéticas reservam geradores inteiros para
seleção e calibração. As reais com fonte `n/a` são distribuídas
deterministicamente por imagem (70/15/15) e recebem
`group_status=real_source_unavailable`; **não** se deve apresentar essa
validação como teste de fonte real inédita. Uma eventual quase-duplicata não é
descartada automaticamente; a checagem perceptual continua uma limitação.

O treino usa somente treino/seleção/calibração internos. O teste OpenFake
histórico é lido para conferir separação por hash, **não é inferido**. O
benchmark RAISE/MNW tampouco é inferido. AUC de seleção decide o melhor
checkpoint; temperatura e limiar usam a calibração interna. O arquivo
`internal_metrics.json` não representa acurácia externa. `last.pt` permite
retomada após uma época fechada e `training_complete.json` impede repetição.

### O que o detector devolve

A rede produz dois *logits* por imagem. Em inferência, a média dos logits dos
cinco recortes é dividida pela temperatura salva no checkpoint e passa por
`softmax`: o valor da classe 1 é `p(sintética)` e o da classe 0 é
`p(real) = 1 - p(sintética)`. Só depois o limiar decide o rótulo:
`p(sintética) >= 0,445` significa "sintética" para o checkpoint atual.
O diagnóstico salva essa probabilidade por imagem em `predictions.csv`.

Na futura API web, a resposta deverá incluir **probabilidade sintética e real,
rótulo, limiar e versão do checkpoint**. O rótulo é uma decisão operacional;
a probabilidade é estimada para a distribuição usada na calibração interna,
sem garantia de calibração em toda foto nova enviada por usuários. "Real"
significa classe fotográfica do dataset, não veracidade da cena ou notícia.

## Passo a passo no Colab

Antes: publique os arquivos novos na branch `main` **somente após
autorização para commit/push**. A sessão do VS Code não vê arquivos locais no
disco E:. Abra `Notebooks/fase_3/fase3_5/rodar_fase3_5_colab.ipynb` e conecte GPU.

1. Execute as células **1–5**: GPU/dependências, Drive, senha da credencial
   GitHub existente, snapshot da branch e comandos. Não refaça a configuração
   inicial de token. Confirme o commit sincronizado e espaço livre.
2. Execute **6, PILOTO MANUAL**. Ele baixa os shards definidos em `SHARDS` para
   `/content` e informa contagens por classe e grupos elegíveis (20+ imagens).
   A saída recebida mostrou candidatos suficientes para 1.500 por classe,
   mas não fonte individual para as reais. O código foi corrigido para
   registrar essa limitação e fazer divisão por imagem somente para esse caso.
   Antes de continuar, sincronize a versão publicada dessa correção. Se um
   piloto futuro não fornecer candidatos suficientes, escolha outros shards
   com metadados internos, nunca com resultados externos.
3. Só com o piloto aprovado e espaço suficiente, execute **7, PREPARAÇÃO
   MANUAL**. Ela persiste uma seleção imutável e faz a verificação de hashes.
   Se cair, repita a mesma célula com a **mesma lista SHARDS**; não troque a
   lista após `plan.json` existir. Não apague `.incompleto/`.
4. Execute **8, CÓPIA MANUAL**. Em cada nova sessão, ela copia/verifica a amostra
   do Drive para `/content` antes de qualquer treino.
5. Execute **9, SMOKE MANUAL**. Deve mostrar contagens, hashes, pesos iniciais
   da Fase 3.3, forward finito e `smoke.json`. Se falhar, não rode o treino.
6. Execute **10, TREINO MANUAL** somente após validar o smoke. A chamada `auto`
   inicia ou retoma a execução no diretório novo. Acompanhe épocas, AUC de
   seleção, perdas e `last.pt`; não considere avisos de processo ativo como
   prova de progresso do modelo.
7. Execute **11, CONFERÊNCIA**. Envie `internal_metrics.json`, última época do
   `history.csv`, status de `last.pt`, `best.pt`, `phase3_final.pt` e qualquer
   erro. Ainda será necessária uma avaliação externa **nova e previamente
   reservada** para testar a meta de 85% sem reutilizar feedback conhecido.

Os comandos equivalentes no terminal do Colab, **após** montar o Drive,
sincronizar o código e executar `stage`, são:

```bash
TCC_CODE_ROOT=/content/tcc_codigo/<sha> python -u /content/tcc_codigo/<sha>/Notebooks/fase_3/fase3_5/fase3_5_finetune.py smoke
TCC_CODE_ROOT=/content/tcc_codigo/<sha> python -u /content/tcc_codigo/<sha>/Notebooks/fase_3/fase3_5/fase3_5_finetune.py auto
```

Substitua `<sha>` pelo commit que a célula 4 imprimiu. `auto` inicia ou retoma;
se `training_complete.json` existir, apenas informa que o treino terminou.

## Estado informado pelas saídas do Colab

O piloto baixou os dois shards para `/content` (1.384.257.833 bytes no total).
O aviso de acesso não autenticado ao HF Hub não impediu o download; o dataset
é público e o token não foi necessário. O preparo congelou 3.000 imagens
internas em `dados_fase3_5/fase3_5_novos_v1/`. O smoke conferiu hashes,
carregou os pesos iniciais da Fase 3.3 e terminou com `Smoke OK`.

O ajuste fino terminou as 12 épocas; o melhor checkpoint foi o da época 10
(AUC de seleção **0,9487973**). A calibração interna retornou temperatura
**0,4409471**, limiar **0,445** e acurácia balanceada **aparente** de
**0,9212768** na própria partição usada para escolher o limiar. As partições
foram 3.594 treino, 719 seleção e 719 calibração; 1.947 exemplos do teste
histórico OpenFake foram reservados e não avaliados. A conferência posterior
listou todos os artefatos esperados, inclusive `training_complete.json`.
Essas verificações são baseadas nas saídas fornecidas pelo usuário; o Drive
não foi montado nesta sessão local. A avaliação externa posterior no
Mirage-Test obteve **60,45%** de acurácia; consulte
`docs/BENCHMARK_EXTERNO_MIRAGE_2000.md`. Este resultado não altera as
métricas internas registradas acima.
Não repetir a célula 10 para prolongar o treino: qualquer variação de épocas
ou outros parâmetros exige um experimento separado e comparação interna.

O diagnóstico interno informado pelo usuário concluiu na GPU Tesla T4 com
719 imagens da seleção: acurácia **0,9068**, acurácia balanceada **0,9080**,
precisão **0,9314**, recall sintético **0,8959**, especificidade real **0,9200**
e AUC **0,9696** em cinco recortes, com TN=299, FP=26, FN=41, TP=353.
Resultado em `resultados/diagnostico_interno_fase3_5_v1/`, segundo a saída
do Colab; não é teste independente. Calibração e conjuntos externos não foram
inferidos nessa célula.

## Diagnóstico interno após o treino

`Notebooks/fase_3/fase3_5/diagnostico_interno_fase3_5.py` confere que o manifesto e as
partições ainda correspondem à execução concluída, carrega somente o checkpoint
final da 3.5 e faz inferência na partição interna de seleção.
Usa cinco recortes e a temperatura e o limiar **congelados** no checkpoint; não
ajusta nenhum parâmetro. Mostra progresso por bloco e salva `summary.json`,
`per_group.csv` e `predictions.csv` em
`resultados/diagnostico_interno_fase3_5_v1/`, sem alterar modelos anteriores.
A calibração, o teste OpenFake reservado e RAISE/MNW não são inferidos. A AUC de
seleção obtida com cinco recortes poderá diferir da AUC registrada durante o
treino, que usou recorte central. A seleção já escolheu o checkpoint; seu
diagnóstico é descritivo e não uma avaliação independente. Grupos pequenos
devem ser lidos com cautela.

Depois de publicar o script com autorização, no notebook da 3.5 execute as
células 4 e 5 para atualizar o snapshot e a **célula 12** para o diagnóstico.
Em sessão nova, execute 1–5 e 8 antes da 12. Não repita preparo ou treino.

O diagnóstico foi concluído no Colab conforme saída fornecida pelo usuário.
Para avançar sem novo treino especulativo, `Notebooks/fase_3/fase3_5/auditar_erros_internos_fase3_5.py`
analisa as predições já salvas da seleção, lê apenas os metadados e dimensões
dessas imagens e grava `errors.csv`, `segments.csv`, `selection_metadata.csv`
e `summary.json` em `resultados/auditoria_erros_internos_fase3_5_v1/`.
A **célula 13** do notebook executa essa auditoria manualmente após sincronizar
o script publicado; em sessão nova execute 1–5 e 8 antes dela. Ela não infere
novamente o modelo, não calibra e não usa conjuntos externos. Consulte
`docs/PLANO_ACURACIA_85.md` para a decisão experimental seguinte.

A auditoria foi concluída conforme saída do Colab: 44 erros entre 537 imagens
Community Forensics e 23 entre 182 imagens OpenFake. A **célula 14** mostrou
um cruzamento de classe, origem e tamanho e seis FP e seis FN internos de
maior confiança. A variante isolada 3.6, que aumentou o redimensionamento,
foi treinada depois; não demonstrou ganho na AUC de seleção interna. O próximo
passo está em `docs/BENCHMARK_EXTERNO_MIRAGE_2000.md`.
