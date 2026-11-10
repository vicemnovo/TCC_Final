# Fechamento documental da Fase 3.9

## O que o notebook reúne

[`fase3_9_final_colab.ipynb`](../Notebooks/fase_3/fase3_9_final_colab.ipynb)
é o notebook único para o trabalho restante da 3.9. Ele **junta** a
preparação do runtime, a cópia local dos dados e a leitura dos resultados de
`rodar_fase3_9_tgif_colab.ipynb` com as perguntas V1, V2 e V3 de
`validar_perguntas_fase3_9_colab.ipynb`. Os dois originais permanecem no
projeto. O notebook final não traz comandos para repetir preparo TGIF,
smoke, treino, calibração ou inferência Mirage já concluídos.

V1 lê configuração, scripts e resultados existentes. V2 examina a estrutura
TGIF, os pares, formatos e qualidade JPEG estimada. V3 procura duplicatas
exatas e candidatos visuais nas divisões e ajusta **somente** um classificador
estatístico de dimensões/bytes usando o treino interno; ele serve para
diagnosticar atalhos e não substitui o detector neural. V3 lê imagens do
teste OpenFake reservado apenas para hashes e tamanho; não infere o modelo
nem usa esse teste para escolher parâmetros. O Mirage é lido somente pela
célula de status e pelo fechamento, sem nova avaliação.

## Execução no Google Colab

Abra `Notebooks/fase_3/fase3_9_final_colab.ipynb` no VS Code e conecte o
runtime Colab da conta correta. **GPU não é necessária.** Execute as células
de código pelos números escritos no início de cada célula:

1. **1–5, preparação:** verificam dependências, montam o Drive, pedem a
   senha da credencial GitHub cifrada e baixam o snapshot do commit atual
   da branch `fase3-ajuste-fino` para `/content`. Nenhum token é impresso.
2. **6, espelho:** copia este notebook e todos os `.ipynb`/`.py` da branch
   para as respectivas pastas em `/content/drive/MyDrive/TCC_Final`.
   Arquivos divergentes já presentes no Drive são preservados em
   `codigo_historico/`; o manifesto fica em `codigo_sync/<commit>.json`.
3. **7 e 8, conferência e V1:** leem a conclusão do treino, o bloqueio e o
   resultado comparativo Mirage já gravados; V1 mostra as perguntas de
   configuração/código. Não ajustam modelo, temperatura ou limiar.
4. **9, cópia local:** traz e verifica os dados internos congelados das
   fases 3.5, 3.7, 3.8 e TGIF 3.9 para `/content`. É necessária antes de
   V2/V3 em cada nova sessão. Pode demorar; mostra progresso periódico.
5. **10, V2:** examina TGIF e formato/compressão. A listagem de arquivos
   `.tar.gz`, se existirem na sessão, pode levar minutos e mostra aviso a
   cada 30 segundos. Grava um novo
   `resultados/validacao_perguntas_3_9_v1/v2_tgif_e_formato_<data-hora-UTC>.json`.
6. **11, V3:** calcula hashes/dHash, identifica possíveis duplicatas e mede
   AUC de um baseline de tamanho nas partições internas. Pode demorar pela
   leitura de 12.475 imagens e informa progresso. Grava um novo
   `v3_duplicatas_e_baseline_<data-hora-UTC>.json` no mesmo diretório.
7. **12, fechamento:** confere que os relatórios desta sessão existem,
   valida SHA-256 do checkpoint contra o bloqueio e o resultado Mirage,
   confere temperatura/limiar congelados e grava um novo
   `fechamento_fase3_9_<data-hora-UTC>.json`. Leia os avisos antes de
   declarar a revisão documental encerrada.

Os três relatórios usam hora **UTC com microssegundos** e abertura exclusiva:
reexecutar V2, V3 ou fechamento cria outro arquivo e não substitui um resumo
antigo. A primeira cópia para o Drive exige que este notebook já esteja
publicado na branch; o próprio código verifica isso. Não execute as células
6, 8, 9 ou 11–13 do notebook histórico de treino para esta revisão.

## Critério de encerramento e limite do resultado

O treino interno e a avaliação comparativa Mirage da 3.9 já terminaram.
O trabalho documental restante é executar V2/V3 até o fim e revisar o
fechamento: se aparecerem arquivos ausentes, pares não identificados,
duplicatas cruzadas ou candidatos visuais, registre e investigue antes de
afirmar que a auditoria está resolvida. Nenhum aviso autoriza alterar a
configuração da 3.9 usando o Mirage. A acurácia comparativa Mirage de 58,75%
fica abaixo da meta; o conjunto já observado não é um teste final cego da
próxima variante. O projeto continuará com uma nova hipótese interna na
Fase 3.9.1 após o fechamento da 3.9.
