# Resultados verificados e decisão após a Fase 3.4 v2

Os valores abaixo vêm de `modelos/*/final_metrics.json` (Fases 2–3.3),
da saída salva de `Notebooks/fase_1/fase1_vit.ipynb` (Fase 1) e das saídas
das células 8–9 da Fase 3.4 v1/v2 fornecidas pelo usuário. Os checkpoints
v1/v2 estão no Drive; suas métricas não foram reexecutadas nesta sessão local.

| Etapa | Conjunto avaliado | Acurácia | AUC | Precisão | Recall | F1 | Especificidade |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fase 1 | Teste próprio faces/COCO | 88,14% | — | 83,96% | 94,31% | 88,83% | — |
| Fase 2 | Teste próprio | 99,74% | — | 99,54% | 99,94% | 99,74% | 99,54% |
| Fase 2 sem ajuste | Referência OpenFake da 3.3 | — | 0,5617 | — | — | — | — |
| Fase 3.1 | OpenFake; holdout diferente | 69,85% | 0,7832 | 71,23% | 63,78% | 67,30% | 75,60% |
| Fase 3.2 | OpenFake | 71,24% | 0,7993 | 75,36% | 60,72% | 67,25% | 81,20% |
| Fase 3.3 | OpenFake | 71,14% | 0,8041 | 77,62% | 57,13% | 65,82% | 84,40% |
| Fase 3.4 v1 | OpenFake | 71,55% | 0,8165 | 77,87% | 57,97% | 66,46% | 84,40% |
| Fase 3.4 v2 | OpenFake | **72,88%** | **0,8222** | **78,05%** | 61,56% | **68,83%** | 83,60% |

O teste próprio da Fase 1 e o teste próprio da Fase 2 **não são comparáveis**
ao OpenFake. A Fase 3.1 usa geradores reservados diferentes. As Fases 3.4 v1
e v2 usam o mesmo manifesto, mas a seleção de replay efêmero da v1 não foi
persistida; a comparação não isola rigorosamente o efeito do sampler.

A v2 identificou 34 imagens sintéticas a mais do que a v1 no OpenFake,
com oito falsos positivos adicionais: TN 836, FP 164, FN 364, TP 583.
A acurácia melhorou 1,34 ponto percentual. O AUC de seleção interna da v2
foi 0,9671 e o AUC OpenFake, 0,8222: há uma lacuna de generalização.

## Decisão pré-declarada para depois da avaliação externa

- **Primeiro:** confirmar integridade dos 20 arquivos e executar uma única
  avaliação congelada da v2. Até essa execução, **não há resultado externo**.
  O resultado não deve alterar checkpoint, temperatura ou limiar.
- **Se o desempenho externo for suficiente para a demonstração:** preservar
  a v2, documentar a incerteza de apenas 20 casos e avançar para a API web.
  Ainda é recomendável obter um conjunto adicional maior e independente para
  sustentar uma afirmação ampla de robustez.
- **Se for insuficiente:** não ajustar a v2 com essas 20 imagens. Investigar
  primeiro os dados **internos**: grupos sintéticos com pouquíssimas imagens,
  só duas fontes reais no treino preparado, formatos e resoluções. A escolha
  entre ampliar o OpenFake ou introduzir outro dataset deve se apoiar nessa
  auditoria e na disponibilidade de dados/licenças, não na otimização das 20
  respostas externas. Se um novo modelo for treinado, essas 20 imagens já
  terão servido como feedback de pesquisa e não poderão ser apresentadas como
  teste final independente desse novo modelo; reservar outro conjunto externo.

O limiar de 85% sobre 20 imagens equivale a 17 acertos. Cada imagem altera a
acurácia observada em cinco pontos percentuais; registrar também a matriz de
confusão, AUC e intervalo de Wilson em vez de apresentar apenas a acurácia.
