# Fase 3.7 — expansão dos dados internos

**Estado:** treino interno e avaliação comparativa no Mirage concluídos
conforme saídas do Colab informadas pelo grupo: acurácia de 62,3% e AUC de
0,6908315. O protocolo abaixo registra a hipótese e as instruções originais;
não é o próximo treino a executar. Veja `ESTADO_PROJETO.md` para a 3.8.

## Hipótese e limite

A 3.7 acrescenta uma seleção de `HFCF_small_1.parquet` e
`HFCF_small_118.parquet`, dois arquivos ainda não usados do
[Community Forensics–Small](https://huggingface.co/datasets/OwensLab/CommunityForensics-Small),
à seleção da 3.5. O piloto deve confirmar se o primeiro fornece imagens
sintéticas, o segundo reais, e se há grupos suficientes. **Não afirmar essas
contagens antes da saída do Colab.** Os arquivos são baixados por inteiro para
`/content`; somente a seleção fixa de 1.500 por classe é persistida no Drive.
O espaço de download esperado é da ordem dos dois shards usados na 3.5
(aproximadamente 1,4 GB), mas deve ser conferido no piloto. A licença informada
pela fonte é CC BY-NC-SA 4.0; o uso aqui é experimental e não comercial.

Isto expande uma fonte interna existente; **não introduz um dataset independente**.
A melhora de generalização é hipótese, não resultado. Arquitetura ViT-Tiny + FFT
+ EfficientNet-B0, checkpoint inicial da 3.3, amostrador por raiz da frequência
do grupo, aumentos moderados, 12 épocas e paciência 3 seguem a 3.5. A 3.6
permanece preservada e não é alterada. As imagens novas são convertidas para
JPEG RGB qualidade 90, como na 3.5, para reduzir uma pista trivial de formato.
As reais sem fonte individual são divididas por imagem, uma limitação explícita.

## Isolamento e artefatos

`experimentos/fase3_7_dados.py` fixa o commit da fonte, os shards, os índices,
grupos, divisões e hashes em `dados_fase3_7/fase3_7_cfsmall_expansao_v1/` no
Drive. A seleção consulta hashes de OpenFake, da 3.5, RAISE/MNW, Mirage e das
imagens históricas somente para impedir duplicatas exatas. **Não lê predições
externas**. O mesmo gerador sintético da 3.5 recebe a mesma divisão na 3.7.
Não há garantia de ausência de quase duplicatas nem de fontes reais inéditas.

`experimentos/fase3_7_finetune.py` usa dados internos OpenFake e os dois
manifestos Community Forensics. O teste OpenFake é verificado por hash, sem
inferência. O melhor checkpoint é escolhido pela AUC da seleção interna;
temperatura e limiar são ajustados somente na calibração interna. O treino salva
`last.pt` para retomada, `best.pt`, `phase3_final.pt`, histórico, configuração,
divisões e métricas em `modelos/fase3_7_cfsmall_expansao_v1/`.

`experimentos/avaliar_fase3_7_mirage.py` congela checkpoint, código, manifesto,
temperatura e limiar antes de inferir as mesmas 2.000 imagens Mirage. O resultado
vai para `resultados/benchmark_externo_mirage_2000_v1/fase3_7_cfsmall_expansao_v1/`.
O Mirage **já foi observado** nas fases anteriores; é comparação, não teste
cego independente. Nunca usar suas respostas para treinar ou recalibrar a 3.7.

## Execução manual no Colab

Abra `Notebooks/fase_3/rodar_fase3_7_colab.ipynb` no VS Code e conecte uma GPU
Colab. As células não iniciam treino ao abrir.

1. **1–5 — ambiente:** dependências, Drive, senha da credencial GitHub existente,
   snapshot publicado da branch e comando com indicador de atividade.
2. **6 — piloto:** baixa os dois shards completos e mostra contagens e grupos;
   não seleciona imagens nem treina. Avance somente se houver classes e grupos
   suficientes. Se falhar, envie a saída; não troque shards de um plano já
   persistido.
3. **7 — preparo/congelamento:** seleciona imagens, reserva divisões, normaliza
   e grava hashes no Drive. Pode retomar um plano incompleto após desconexão.
4. **8 — cópia local:** copia as seleções 3.5 e 3.7 para `/content`, conferindo
   hashes. Repita em cada nova sessão.
5. **9 — smoke test:** verifica arquitetura, pesos iniciais, separação das
   divisões, ausência de duplicatas exatas, forward na GPU e gravação. Não
   comprova acurácia. Avance somente com `Smoke OK`.
6. **10 — treino:** execute manualmente para iniciar ou retomar de `last.pt`.
   Acompanhe loss, AUC e acurácia balanceada da seleção, melhor época e
   `training_complete.json`. A existência desse arquivo indica treino interno
   concluído, **não** avaliação externa concluída.
7. **11 — conferência interna:** leia os artefatos e a calibração, sem escolher
   parâmetros pelo Mirage.
8. **12–15 — avaliação externa comparativa:** copie Mirage, congele o modelo,
   avalie uma vez e leia acurácia, AUC, recall sintético, especificidade real,
   matriz de confusão e comparação pareada com a 3.5.

Para retomar após desconexão: conecte GPU, execute 1–5, execute 8 e depois
10. Se o preparo não terminou, execute 7 antes de 8. Não repita 9 caso o smoke
já tenha sido salvo. Não execute 10 quando `training_complete.json` existir.

O piloto Open Images preparado anteriormente permanece opcional e separado.
Nenhum metadado de origem é exigido na inferência de imagens de usuários.
