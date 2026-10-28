# Avaliação externa histórica da Fase 3.4 v2 — 20 imagens

Esta avaliação já ocorreu pelo método antigo de 20 imagens, com origem não
documentada. Seus resultados são preservados. O novo protocolo de 2.000 imagens
RAISE + MNW está descrito em `docs/BENCHMARK_EXTERNO_2000.md` e ainda não foi
executado; ele não altera as métricas desta página.

**Estado:** execução única concluída no Colab conforme a saída fornecida pelo
usuário. Não repetir a inferência nem usar este conjunto para ajustes.

## Resultado registrado

Em 20 imagens (8 reais e 12 sintéticas), a v2 acertou 11: **acurácia 55%**,
intervalo de Wilson de 95% **[34,21%, 74,18%]**, AUC **0,6875**, acurácia
balanceada **62,5%**, precisão **100%**, recall sintético **25%**, F1 **40%**
e especificidade **100%**. Matriz de confusão: TN=8, FP=0, FN=9, TP=3.
O alvo de 85% corresponderia a 17 acertos; não foi atingido nesta execução.
O valor de precisão corresponde a apenas três predições positivas corretas,
portanto não deve ser apresentado isoladamente como prova de robustez.

O Colab registrou GPU Tesla T4, 20 hashes externos distintos, nenhuma
sobreposição **exata** com as 3.979 imagens das partições OpenFake, carga dos
pesos finais da v2 e inferência com cinco recortes. Temperatura e limiar foram
os valores congelados abaixo. Saídas persistidas, conforme o log:
`/content/drive/MyDrive/TCC_Final/resultados/avaliacao_externa_fase3_4_v2_v1/metrics.json`
e `predictions.csv`. O Drive não foi montado nesta sessão local; os valores
acima provêm da saída do usuário, não de uma reavaliação local.

Hashes de referência registrados no resultado: checkpoint
`fbdb11314056d6ccae338fc8f117e80cb7017eaa6581073cab9defed5ca8a6ab`;
rótulos `14e73e8184f49e0fb0b6530f39b7310426ea4a31cc1f601a752b679b0425d2a3`;
partições `a8a59ba2681249c9f24b2ea46b2adaeabe8f091f7c0343def238ccf633c131de`.

O candidato está congelado: checkpoint
`modelos/fase3_4_finetune_v2_sqrt_sampler/phase3_final.pt`, temperatura
`0,44025686383247375` e limiar `0,36999999999999994`, conforme a saída da
célula 9. O script lê esses valores do checkpoint e exige concordância com
`final_metrics.json`; não os recalcula ou ajusta.

O conjunto é `validacao_fase3/teste_externo/rotulos.csv`, com 20 imagens
distintas: 8 reais e 12 sintéticas. Seus nomes diferem daqueles em
`teste_externo/external_predictions.csv`, um artefato histórico que **não é
entrada desta avaliação** e não será sobrescrito. Cópias desses 20 arquivos
existem em outras pastas; somente os caminhos do CSV de rótulos são usados.
O campo `origem` dos 20 rótulos está como `desconhecido`; checar ausência de
duplicatas exatas não comprova independência de origem ou de conteúdo.

## Procedimento usado no Colab (registro; não repetir)

1. Publique o código novo na branch `fase3-ajuste-fino` **somente após
   autorização para commit/push**. O código local não aparece no snapshot
   remoto sem publicação. Não mova os dados nem os checkpoints.
2. No notebook `Notebooks/fase_3/rodar_fase3_4_v2_colab.ipynb`, conecte GPU
   e execute as células 1–5 para montar Drive, desbloquear credencial e
   sincronizar o snapshot atualizado. A célula 6 apenas confirma o manifesto.
   **Não repita as células 7 ou 8**: smoke e treino já concluíram.
3. Execute a célula **10**, explicitamente marcada como avaliação externa
   final manual. Ela chama `experimentos/avaliar_externo_fase3_4_v2.py`.
   A verificação de hashes de 3.979 arquivos OpenFake pode levar muitos
   minutos; o processo informa avanço periodicamente. O script bloqueia
   sobreposição exata com treino, seleção, calibração e teste OpenFake.
4. A saída da célula 10 apontou os arquivos em
   `/content/drive/MyDrive/TCC_Final/resultados/avaliacao_externa_fase3_4_v2_v1/`:
   `metrics.json` e `predictions.csv`. A pasta é exclusiva; uma avaliação
   anterior não é substituída. **A execução já terminou**: não rode a célula
   de novo para tentar melhorar o resultado.

O modelo usa os mesmos cinco recortes e a mesma normalização da avaliação
OpenFake. As imagens externas originais são lidas sem recompressão adicional.
São reportados acurácia, intervalo de Wilson para a acurácia, AUC, precisão,
recall, F1, especificidade e matriz de confusão. Com apenas 20 imagens, cada
acerto muda a acurácia em cinco pontos percentuais: o resultado é útil para
diagnóstico, mas tem incerteza amostral grande. A avaliação não inicia treino,
não calibra, não escolhe limiar e não altera artefatos anteriores.

Depois desta avaliação, decidir entre continuar com OpenFake ou introduzir
outro dataset **não transforma o resultado dessas mesmas 20 imagens em teste
independente para o próximo modelo**. Qualquer novo modelo precisará de um
novo conjunto externo reservado para uma avaliação final honesta.
