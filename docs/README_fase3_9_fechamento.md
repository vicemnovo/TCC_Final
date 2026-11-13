# Fechamento documental da Fase 3.9

**As duas conferências foram executadas; para reproduzi-las, use**
[`conferir_fase3_9_colab.ipynb`](../Notebooks/fase_3/conferir_fase3_9_colab.ipynb).
Ele é um arquivo novo e independente, com três células de código identificadas
como passos 1, 2 e 3. Abra a cópia **local do repositório** no VS Code, pois a
cópia antiga do notebook final no Drive pode não conter a célula 14.
O passo 2 valida diretamente os 768 pares TGIF; o passo 3 cria galerias sem
abrir imagens do teste reservado. O notebook não precisa de token nem GPU.
Se a lista `achados` da V3 ainda estiver na memória do mesmo kernel, a galeria
cobre todos os candidatos internos. Em outro kernel, usa os até 30 candidatos
mais próximos que ficaram impressos no log e informa a cobertura parcial.
O grupo executou esse modo parcial e enviou `resumo.json`, as três galerias
e `candidatos.csv`. Os 19 pares exibidos foram inspecionados: dois são
fotografias visualmente repetidas envolvendo calibração, um reutiliza a
mesma cena com uma alteração e os demais parecem distintos. A revisão e
os nomes exatos das divisões estão em
[`REVISAO_VISUAL_FASE3_9.md`](REVISAO_VISUAL_FASE3_9.md).

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
7. **12, fechamento:** confere V2 e V3 (pode reutilizar o V3 já gravado),
   valida SHA-256 do checkpoint contra o bloqueio e o resultado Mirage,
   confere temperatura/limiar congelados e grava um novo
   `fechamento_fase3_9_<data-hora-UTC>.json`. Leia os avisos antes de
   declarar a revisão documental encerrada.
8. **13, log:** arquiva no Drive as saídas textuais já executadas e anexa
   os JSON V2, V3 e fechamento desta revisão. Cria um arquivo datado em
   `resultados/validacao_perguntas_3_9_v1/logs/`, sem substituir logs antigos.
9. **14, conferência visual:** verifica diretamente os 768 pares do manifesto
   TGIF e usa os candidatos da V3 ainda em memória para criar CSV e galerias
   datadas em `resultados/validacao_perguntas_3_9_v1/revisao_visual_interna_*`.
   Mostra a primeira galeria no notebook. Não abre imagens do teste reservado
   nessa revisão visual; a interpretação dos pares depende de inspeção humana.
   Execute na mesma sessão em que a célula 11 produziu `achados`. Se o runtime
   foi reiniciado, a lista completa de candidatos precisa ser reconstruída
   pela célula 11; o JSON V3 anterior guarda contagens, não as identidades.

Os relatórios e o log usam hora **UTC com microssegundos** e abertura exclusiva:
reexecutar V2, V3 ou fechamento cria outro arquivo e não substitui um resumo
antigo. A primeira cópia para o Drive exige que este notebook já esteja
publicado na branch; o próprio código verifica isso. Não execute as células
6, 8, 9 ou 11–13 do notebook histórico de treino para esta revisão.

## Critério de encerramento e limite do resultado

O treino interno e a avaliação comparativa Mirage da 3.9 já terminaram.
As células 1–12 foram executadas em 10/11/2026. V2, V3 e fechamento estão
gravados, mas o fechamento apontou revisão técnica pendente. A V2 anterior
não identificou os pares porque procurava `path`, enquanto o manifesto TGIF
usa `rows` com `image` e `image_id`. Isso foi corrigido. Após publicar o
notebook atualizado, execute **1–7, 9, 10, 12 e 13**. A célula 6 espelha o
novo notebook e o exportador de log no Drive. A V3 já verificou
12.475 imagens em cerca de 38 minutos; a célula 12 pode reutilizar o relatório
V3 mais recente, sem repetir essa leitura. A célula 13 lê as saídas que foram
salvas no snapshot versionado do notebook. Se o runtime ainda usa um commit
antigo, **reexecute as células 3 e 4**, pois a 4 apaga o token após cada
sincronização. Depois execute 6, 10, 12 e 13 para espelhar o código, corrigir
V2, fechar a revisão e criar o log. O erro
`python3: can't open file .../exportar_log_notebook.py` com código 2
identifica precisamente esse snapshot antigo, pois o script só entrou no
commit `ee8605d`.

A V3 achou três duplicatas exatas somente dentro do teste reservado, nenhuma
entre divisões e nenhum rótulo conflitante. Há 71 candidatos visuais por
dHash; um par entre treino e seleção tem MAE RGB 1,25 e precisa de inspeção
visual. Esses números não comprovam vazamento, mas a auditoria visual ainda
não pode ser declarada concluída. Nenhum aviso autoriza alterar a
configuração da 3.9 usando o Mirage. A acurácia comparativa Mirage de 58,75%
fica abaixo da meta; o conjunto já observado não é um teste final cego da
próxima variante. O projeto continuará com uma nova hipótese interna na
Fase 3.9.1 após o fechamento da 3.9.
