# Fase 3.8 — piloto de dados AIGI pareados

**Estado histórico do piloto:** o preparo posterior congelou 480 pares AIGI,
a cópia local foi conferida e o smoke test passou no Colab. Para continuar o
experimento, use `README_fase3_8_execucao.md` e `ESTADO_PROJETO.md` na raiz;
as instruções de piloto abaixo documentam a etapa anterior.

## Função experimental da versão

A auditoria interna da 3.7 encontrou, em seus **3.000 novos exemplos**, reais
originalmente JPEG com lado curto mediano de 428 px e sintéticas originalmente
PNG com lado curto de 512 px. A conversão posterior de ambas para JPEG não
garante remover o histórico distinto de compressão. Isso é uma associação de
dados; ainda não demonstra que a rede use esse atalho. A 3.7 obteve 62,3% de
acurácia no Mirage já observado e continua preservada.

A hipótese da **3.8** é treinar com fotografias e imagens sintéticas de cenas
correspondentes, variedade de geradores e uma distribuição interna de tamanho
e processamento menos associada à classe. A fonte candidata é
[AIGI-Detection-Quality-Paradox](https://huggingface.co/datasets/Coxy7/AIGI-Detection-Quality-Paradox),
criada para o [estudo ICML 2025](https://openreview.net/forum?id=sKYdVKE1tS).
Ela informa 24.000 sintéticas de seis geradores, ligadas por `uid` a 4.000
fotografias usadas como referência de conteúdo. **As fotos reais não estão
incluídas no repositório**: há somente um CSV com suas fontes. A licença CC BY
4.0 da publicação AIGI não substitui os termos de cada fonte real.

O objetivo imediato é verificar a viabilidade das fotografias pareadas. Não
adicionar ViT, CNN ou FFT, mudar limiar, número de épocas ou montar treino com
sintéticas isoladas. Se os pares forem insuficientes, parar e trocar a fonte
real antes de qualquer seleção. Rastreabilidade no CSV serve para auditar
rótulos e separar divisões; a rede continuará a receber **apenas pixels**.

## Piloto implementado

- `Notebooks/fase_3/fase3_8/fase3_8_aigi_piloto.py`: fixa o commit público do dataset;
  baixa por inteiro `real_images_metadata.csv` (cerca de 1,8 MB) e
  `data/fake-00000-of-00015.parquet` (cerca de 495 MB) para `/content`;
  verifica esquema, rótulos implícitos, geradores, formatos e resoluções;
  confere a leitura de uma sintética; procura `uid` pareados com URLs HTTPS
  das fontes LAION/CC3M; baixa por inteiro até 16 fotos reais para verificar
  disponibilidade, formato e tamanho.
- `Notebooks/fase_3/fase3_8/piloto_fase3_8_aigi_colab.ipynb`: interface Colab com
  credencial GitHub já configurada, snapshot da branch e indicador de atividade.
  CPU basta. Nenhuma célula treina ao abrir.
- Saída temporária: `/content/fase3_8_aigi_piloto/pilot_summary.json`. Não
  há manifesto de treino, checkpoint ou resultado externo da 3.8 nesta etapa.

O piloto usa URLs reais públicas fornecidas pela fonte. URLs indisponíveis são
contadas sem imprimir seu conteúdo; sucesso de 16 tentativas não garante a
disponibilidade das demais. Os dados são examinados **após** o download
completo dos arquivos selecionados, sem `streaming=True`.

## Como executar agora

1. Abra `Notebooks/fase_3/fase3_8/piloto_fase3_8_aigi_colab.ipynb` no VS Code e conecte
   um runtime Colab. Pode ser CPU; não reserve GPU para o piloto.
2. Execute as **células 1–5**: dependências, Drive, senha de desbloqueio da
   credencial GitHub existente, código publicado e comando com progresso.
3. Execute a **célula 6** manualmente. O piloto não cria divisões nem treina.
   A célula pode demorar durante o download do shard e imprime atividade.
4. Execute a **célula 7**. Informe os valores de geradores, formatos, resoluções,
   `paired_https_candidates_in_shard`, `real_probe_success` e
   `real_probe_short_side_512_plus`. Não envie fotos nem credenciais.

## Critério para continuar

Somente depois de observar o piloto, definir quantidade viável de pares,
fontes reais permitidas, geradores, teto de armazenamento e divisão por `uid`
antes de preparar dados. Seleção, calibração e treino devem ter UIDs distintos;
o mesmo conteúdo real e suas gerações não podem atravessar divisões. Auditar
duplicatas com OpenFake, Community Forensics e conjuntos reservados, sem usar
predições externas. Conferir distribuição de resolução, formato e tamanho nas
**duas classes e em todo o treino combinado**, não apenas no novo subconjunto.

Se esses critérios forem atendidos, criar versão separada com preparo
persistente no Drive, cópia local verificada, smoke test sem treino, ajuste fino
com retomada e avaliação Mirage congelada. Seleção interna escolhe a época;
calibração interna define temperatura e limiar. O Mirage já foi observado e
serve apenas para comparação descritiva, não para decisões da 3.8. Para
demonstrar 85% em avaliação final independente, reservar outro conjunto ainda
não observado.
