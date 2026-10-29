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
O fator experimental principal é **adicionar dados internos com geradores e
fontes reais documentados**. O checkpoint inicial continua o da Fase 3.3,
fixado antes da avaliação da 3.5. Não retreina nem altera as Fases 1–3.4.

## Fonte escolhida para o piloto

[Community Forensics–Small](https://huggingface.co/datasets/OwensLab/CommunityForensics-Small)
é publicada pelo grupo responsável pela pesquisa, contém rótulo, nome do
gerador, fonte real e bytes da imagem. A licença informada é **CC-BY-NC-SA 4.0,
para pesquisa não comercial**. Seu uso no TCC experimental não concede
automaticamente permissão para uma API comercial. O conjunto inteiro ocupa
cerca de 260 GB e **não cabe** na quota gratuita; por isso, o piloto baixa
somente arquivos Parquet explicitamente escolhidos, por inteiro, em `/content`.
Não usa `streaming=True`, não baixa o acervo completo e não instala serviço pago.
Os dois shards iniciais do notebook são uma hipótese de coleta, não uma seleção
validada. O piloto deve confirmar classes e grupos antes de `prepare`.

O artigo [Community Forensics](https://arxiv.org/abs/2411.04125) fundamenta
o foco em diversidade de geradores. O estudo
[Fake or JPEG?](https://arxiv.org/abs/2403.17608) alerta para atalhos de
compressão e resolução. Por isso, os dados novos de ambas as classes são
normalizados para JPEG RGB qualidade 90, lado maior limitado a 1024 px quando
possível e lado menor ampliado para pelo menos 224 px; a origem e o hash dos
bytes originais permanecem no manifesto. Isso reduz um atalho simples, mas
não comprova ausência de viés sem auditoria adicional.

## Arquivos e isolamento

- `experimentos/fase3_5_dados.py`: `pilot`, `prepare` e `stage`.
- `experimentos/fase3_5_finetune.py`: `smoke`, `train`, `resume` e `auto`.
- `Notebooks/fase_3/rodar_fase3_5_colab.ipynb`: interface Colab.
- Dados persistentes: `dados_fase3_5/fase3_5_novos_v1/` no Drive; enquanto
  incompleto, `.incompleto/`. Cópia efêmera: `/content/fase3_5_novos_v1/`.
- Novo experimento: `modelos/fase3_5_cfsmall_v1/`; smoke separado em
  `modelos/fase3_5_cfsmall_v1_smoke/`.

`prepare` fixa o commit do dataset, shards, seleção, divisões por grupo e hashes
em `plan.json`/`manifest.json`. Rejeita duplicatas exatas e sobreposição exata
com OpenFake, benchmark RAISE/MNW e imagens históricas localizadas no Drive.
Não examina predições externas. As divisões novas reservam grupos inteiros de
geradores/fontes para seleção e calibração. Uma eventual quase-duplicata não é
descartada automaticamente; a checagem perceptual continua uma limitação.

O treino usa somente treino/seleção/calibração internos. O teste OpenFake
histórico é lido para conferir separação por hash, **não é inferido**. O
benchmark RAISE/MNW tampouco é inferido. AUC de seleção decide o melhor
checkpoint; temperatura e limiar usam a calibração interna. O arquivo
`internal_metrics.json` não representa acurácia externa. `last.pt` permite
retomada após uma época fechada e `training_complete.json` impede repetição.

## Passo a passo no Colab

Antes: publique os arquivos novos na branch `fase3-ajuste-fino` **somente após
autorização para commit/push**. A sessão do VS Code não vê arquivos locais no
disco E:. Abra `Notebooks/fase_3/rodar_fase3_5_colab.ipynb` e conecte GPU.

1. Execute as células **1–5**: GPU/dependências, Drive, senha da credencial
   GitHub existente, snapshot da branch e comandos. Não refaça a configuração
   inicial de token. Confirme o commit sincronizado e espaço livre.
2. Execute **6, PILOTO MANUAL**. Ele baixa os shards definidos em `SHARDS` para
   `/content` e informa contagens por classe e grupos elegíveis (20+ imagens).
   **Pare aqui e confira a saída**: para preparar, cada classe precisa de ao
   menos três grupos elegíveis e de amostras suficientes para a meta de 1.500
   por classe. Se faltar classe/grupo, escolha outros shards com base em
   metadados da fonte, não em resultados externos, e repita somente o piloto.
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
TCC_CODE_ROOT=/content/tcc_codigo/<sha> python -u /content/tcc_codigo/<sha>/experimentos/fase3_5_finetune.py smoke
TCC_CODE_ROOT=/content/tcc_codigo/<sha> python -u /content/tcc_codigo/<sha>/experimentos/fase3_5_finetune.py auto
```

Substitua `<sha>` pelo commit que a célula 4 imprimiu. `auto` inicia ou retoma;
se `training_complete.json` existir, apenas informa que o treino terminou.

## Estado verificado nesta sessão local

Código, notebook e testes locais foram preparados. **Nenhum shard foi baixado,
nenhum manifesto remoto foi criado e nenhum treino 3.5 foi iniciado.** A
composição real dos shards iniciais e a disponibilidade de espaço no Colab
dependem do piloto remoto. Não há métricas da Fase 3.5.
