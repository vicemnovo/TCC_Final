# Fase 3.9: execução TGIF SD2-fr

## O que o piloto comprovou

O piloto TGIF informou 341 IDs compartilhados entre fotografias originais
e versões SD2 totalmente regeneradas na divisão `validation` dos autores.
Os dois arquivos foram baixados integralmente e oito amostras PNG de cada
um foram decodificadas. Nenhum treino ou teste externo foi iniciado. Os
arquivos TGIF2/FLUX tiveram somente o tamanho remoto verificado e não
entram nesta variante.

O CSV oficial `training` contém 2.440 linhas originais, mas apenas 1.558
IDs únicos, pois algumas fotos pertencem a várias categorias. A seleção
da 3.9 é por ID: 768 pares únicos distribuídos em 512 para treino, 128
para seleção da melhor época e 128 para calibração. Real e sintética do
mesmo ID ficam na mesma divisão. A origem das duas classes é processada
para JPEG com lado curto de 256 pixels, reduzindo atalhos de formato e
resolução. Este controle não elimina todos os possíveis vieses.

O primeiro preparo encontrou no tar imagens com o mesmo ID e basename em
categorias diferentes, mas com recortes distintos. A leitura agora usa o
caminho completo `training/<categoria>/<nome>` de cada lado do par. O plano
de IDs já salvo no Drive continua válido e os dois arquivos de treino já
baixados em `/content` são reutilizados na mesma sessão; nenhuma imagem
parcial ou checkpoint anterior é substituído.

## Procedimento no Colab

Abra `Notebooks/fase_3/rodar_fase3_9_tgif_colab.ipynb` e execute:

1. **Células 1–5:** conectam GPU e Drive, desbloqueiam a credencial GitHub
   existente e sincronizam o código da branch publicada. A senha não fica
   no projeto. Nenhum treino começa nessas células.
2. **Célula 6, preparo/congelamento:** baixa os dois arquivos `training`
   completos do TGIF para `/content` (11,56 GB comprimidos), fixa o plano
   e persiste os 768 pares normalizados e seus hashes no Drive. Se cair,
   execute novamente: o download `.part` e o plano podem ser retomados.
   Ela não usa os arquivos `validation`/`testing` do TGIF como treino.
3. **Célula 7, cópia local:** copia dados internos persistidos da 3.5, 3.7,
   3.8 e 3.9 para `/content` com verificação. Repita a cada nova sessão.
4. **Célula 8, smoke test:** confere divisões, duplicatas exatas, GPU,
   arquitetura, checkpoint inicial da 3.3, passagem forward e escrita.
   Não ajusta pesos. Avance somente se aparecer `Smoke OK`.
5. **Célula 9, treino manual:** inicia ou retoma o ajuste fino a partir de
   `last.pt`, em `modelos/fase3_9_tgif_sd2fr_v1/`. Execute explicitamente;
   abrir o notebook não inicia treinamento. A seleção interna escolhe a
   época e a calibração interna fixa temperatura e limiar.
6. **Célula 10:** lê arquivos e métricas internas. Treino interno concluído
   não equivale a avaliação externa concluída.
7. **Células 11–14:** copiam as mesmas 2.000 imagens Mirage congeladas,
   registram checkpoint/temperatura/limiar antes de inferir, executam a
   avaliação comparativa e leem o resultado. O Mirage já foi observado:
   não usar suas respostas para selecionar pesos, dados ou parâmetros.

Os dados preparados ficam em `dados_fase3_9/tgif_sd2fr_768_v1/` no Drive.
Checkpoints e resultados anteriores continuam em seus diretórios próprios.
A 3.9 mantém arquitetura, checkpoint inicial, transformações e parâmetros
da 3.8; a mudança planejada é acrescentar estes pares TGIF internos.

**Estado verificado no notebook em 05/11/2026:** preparo, cópia local,
smoke, treino interno e avaliação Mirage terminaram. Melhor época 10/12, AUC de seleção
0,8988236, temperatura interna 0,9187889 e limiar interno 0,47. Os
arquivos `phase3_final.pt` e `training_complete.json` estão presentes
conforme saída do Colab. As células de código #11 a #14 congelaram os
parâmetros e avaliaram as 2.000 imagens Mirage: acurácia **0,5875**,
AUC **0,6328105**, recall sintético **0,431** e especificidade real **0,744**.
O resultado está em `resultados/benchmark_externo_mirage_2000_v1/fase3_9_tgif_sd2fr_v1/`
no Drive. Não repetir a avaliação nem usar o Mirage para escolher dados,
pesos, limiar ou hiperparâmetros da próxima versão.

Para concluir a revisão das perguntas V1–V3 e registrar o fechamento sem
repetir treino ou avaliação, use o
[`fase3_9_final_colab.ipynb`](../Notebooks/fase_3/fase3_9_final_colab.ipynb).
Ele reúne a preparação deste notebook com a validação do notebook separado;
o procedimento e os relatórios com data/hora estão em
[`README_fase3_9_fechamento.md`](README_fase3_9_fechamento.md).
