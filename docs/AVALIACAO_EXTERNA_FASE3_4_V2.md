# Avaliação externa final da Fase 3.4 v2

**Estado:** preparada localmente; ainda não executada no Colab. Não há
métrica externa final verificada até que esta avaliação rode no runtime remoto.

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

## Como executar no Colab

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
4. Confira a saída da célula 10 e os arquivos em
   `/content/drive/MyDrive/TCC_Final/resultados/avaliacao_externa_fase3_4_v2_v1/`:
   `metrics.json` e `predictions.csv`. A pasta é exclusiva; uma avaliação
   anterior não é substituída. Envie as métricas para a decisão sobre o
   próximo experimento. Não rode a célula de novo para tentar melhorar o
   resultado.

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
