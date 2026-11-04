
# TCC_Final — Instruções para o Codex

## Objetivo
Concluir o projeto de detecção robusta de imagens reais e imagens
sintéticas geradas ou alteradas por inteligência artificial. A aplicação
web com API fica para uma etapa futura; não direcionar o trabalho atual
para ela enquanto a prioridade for melhorar a acurácia do detector.

A detecção de vídeos está fora do escopo atual por prazo e custo;
poderá ser considerada em trabalho futuro, sem direcionar os
experimentos ou entregas desta etapa.

Meta: alcançar pelo menos 85% de acurácia em um conjunto externo
independente, com imagens não utilizadas no treinamento.

A meta é um objetivo experimental, não um resultado já alcançado.
Priorizar a acurácia final e a generalização em imagens externas
independentes em todas as decisões atuais. Não adaptar treino, calibração
ou hiperparâmetros aos resultados de benchmarks externos já observados.

## Tecnologias
- Python e PyTorch
- ViT-Tiny, FFT e EfficientNet-B0
- Google Colab com GPU
- Google Drive para datasets e checkpoints
- Git e GitHub para versionamento

## Caminhos
Projeto local: E:\TCC\TCC_Final
Google Drive no Colab: /content/drive/MyDrive/TCC_Final

## Organização do repositório
- `Notebooks/fase_1/`: notebook original e documentação histórica da Fase 1.
- `Notebooks/fase_2/`: treinamento e teste histórico da Fase 2.
- `Notebooks/fase_3/`: Fases 3.1–3.3 e interfaces Colab das fases atuais.
- `Notebooks/configuracao/`: configuração inicial de credenciais no Colab.
- `experimentos/`: código Python principal dos experimentos; a Fase 3.4 depende
  de `Notebooks/fase_3/fase3.3_compressao.ipynb` no snapshot do GitHub.
- `docs/`: instruções de execução; `README.md` é o índice do projeto.
- `modelos/`, `minhas_imagens/` e `validacao_fase3/` contêm artefatos
  legados. A pasta local `teste_externo/` foi retirada; resultados históricos
  permanecem documentados e cópias das imagens existem nos outros caminhos.
- Não mover dados, checkpoints ou resultados para acompanhar mudanças de código.

## Prioridade atual
A Fase 3.7 foi treinada e avaliada no Mirage: acurácia 0,623 e AUC
0,6908315, conforme saída informada pelo usuário. A Fase 3.8 incorporou
480 pares AIGI internos, com 80 pares por gerador. Preparo, smoke, treino
interno de 12 épocas e avaliação comparativa Mirage da 3.8 foram concluídos
conforme saídas salvas no notebook: melhor época 12, AUC de seleção 0,9603094,
temperatura 0,5229031 e limiar 0,31. No Mirage, a 3.8 teve acurácia 0,599 e
AUC 0,641268, abaixo da 3.7; recall sintético 0,557 e especificidade real
0,641. Preservar ambos os checkpoints e resultados. A próxima ação é auditar
erros e distribuições **internas** para definir uma hipótese de dados em
versão isolada, sem ajustar a 3.8 pelas respostas do Mirage. Detalhes em
`ESTADO_PROJETO.md` e `docs/README_fase3_8_execucao.md`. A auditoria interna
da 3.8 foi preparada em `experimentos/auditoria_interna_fase3_8.py` e
`Notebooks/fase_3/auditar_interno_fase3_8_colab.ipynb`; ainda não foi
executada no Colab. Ela deve inferir somente a seleção interna, preservar
calibração/testes e registrar saída nova. Quatro opções para a 3.9 estão em
`docs/AUDITORIA_E_OPCOES_FASE3_9.md`; nenhuma foi escolhida pelos resultados
externos.

O smoke conferiu 6.152 imagens de treino, 1.420 de seleção, 1.420 de
calibração e 1.947 do teste OpenFake apenas por hash. Os pesos iniciais da
Fase 3.3 foram carregados. A auditoria mostrou que os pares AIGI novos
têm o mesmo formato e faixa de tamanho nas duas classes, mas a associação
classe × resolução continua forte no Community Forensics histórico. Tratar
essa associação como limitação a investigar internamente, sem ajustar a 3.8
pelos resultados externos.

O script de treino 3.8 imprime `Partições 3.7` por mensagem herdada. As
partições efetivas incluem 320 reais e 320 sintéticas AIGI no treino; o
checkpoint e a configuração guardam o hash do script executado. Não alterar
retroativamente esse script apenas para corrigir o texto do log.

A Fase 3.5 foi treinada internamente, teve diagnóstico na seleção e já
foi avaliada uma vez no Mirage-Test externo. Preservar esse checkpoint e
usar apenas a auditoria interna anterior para definir outra variante.
Procedimento em `docs/README_fase3_5.md`.

A Fase 3.6 foi treinada internamente nas mesmas partições da 3.5,
alterando somente a probabilidade de `RandomDownUp` de 0,15 para 0,50.
Sua AUC de seleção foi 0,9472355, contra 0,9487973 da 3.5; não há
evidência interna de melhora. Preservar ambos os checkpoints e não
repetir o treino concluído. Ver `docs/README_fase3_6.md`. Na comparação
externa Mirage informada pelo usuário, a 3.6 obteve acurácia 0,572 e AUC
0,6303885, abaixo da 3.5 (0,6045 e 0,679972). Ambas permanecem abaixo
da meta de 85%; preservar checkpoints e resultados.

A avaliação comparativa da 3.6 no Mirage-Test 2000 v1 foi informada como
concluída. A partir desta fase, cada versão treinada deve ter teste externo
final nas mesmas 1.000
fotos reais e 1.000 sintéticas congeladas do Mirage, com checkpoint,
temperatura e limiar definidos internamente antes de inferir. Registrar
resultados por fase em diretórios separados; nunca substituir resultados
anteriores. Esse protocolo serve para comparação entre fases. Como o Mirage
já foi observado, não chamá-lo de teste cego independente de novas fases,
nem usar suas predições para treinar, calibrar ou escolher parâmetros.

O Mirage-Test 2000 v1 foi informado como congelado e avaliado uma vez com
a Fase 3.5, conforme saída do Colab fornecida pelo usuário: 60,45% de
acurácia, AUC 0,679972 e recall sintético 0,351. Procedimento e limitações
em `docs/BENCHMARK_EXTERNO_MIRAGE_2000.md`. Esse conjunto agora é visto:
não usar suas predições para treinar, calibrar, escolher pesos, dados ou
hiperparâmetros da próxima variante, nem apresentá-lo como teste final
independente dessa variante. Preservar o manifesto, checkpoint e resultados.
A ausência de duplicatas exatas não exclui imagens derivadas da mesma origem.

Foi preparado um piloto opcional **não executado** de Open Images antes da
escolha dos dados da 3.7: `experimentos/fase3_7_openimages_pilot.py`,
`Notebooks/fase_3/piloto_fase3_7_openimages_colab.ipynb` e
`docs/README_fase3_7_piloto.md`. Ele verifica apenas metadados e três JPEGs
de uma fonte real com autoria declarada; não cria divisões nem treina. Antes
de incorporar a fonte, conferir a saída real do piloto, licença, espaço,
proveniência e isolamento por autor. Autoria declarada não prova ausência de
edição por IA. Esse piloto é opcional e não define o objetivo da 3.7:
o detector deve funcionar sem metadados de origem de imagens enviadas por
usuários. Priorizar diversidade de conteúdo real e sintético, qualidade de
rótulos, ausência de duplicatas e redução de atalhos de resolução/formato.
Na Fase 3.7 em diante, tratar ruído e sinais locais de pixel, consistência
de cor/pigmentação, coerência global da cena e sinais de frequência como
hipóteses visuais a testar em dados internos. A expressão "copigmentação"
usada pelo autor será operacionalizada como consistência de cor/pigmentação
até haver definição mais específica. A FFT já calcula o espectro de
magnitude; ViT e CNN não comprovam, por si, raciocínio físico ou explicação
de um artefato. Não declarar que uma pista foi detectada sem uma avaliação
específica. Priorizar dados visualmente diversos e um fator experimental
por versão antes de acrescentar ramos ou complexidade de GPU.
Metadados de origem servem apenas para auditar rótulos e separar divisões
quando disponíveis; não podem ser entrada obrigatória do modelo nem condição
para inferência. Não alterar a 3.5/3.6 por causa do piloto.

Na retomada, ler primeiro `ESTADO_PROJETO.md` e
`docs/GUIA_METODOLOGIA_TCC.md`. Não alterar retroativamente o script de
avaliação Mirage, o notebook de arquitetura da 3.3 ou os checkpoints da 3.5:
seus hashes foram registrados no resultado. Para novas hipóteses, criar
versão isolada, documentar as mudanças e preservar os artefatos anteriores.

Antes de novos treinos, auditar distribuição, formatos, resoluções e
duplicatas dos dados; avaliar generalização por gerador e fonte;
planejar mudanças incrementais de robustez. Adiar a aplicação web e a API
para depois da etapa de melhoria e avaliação final do detector.

Examinar os notebooks e checkpoints existentes antes de
alterar a arquitetura ou iniciar novos experimentos.

## Modo de trabalho
- Pode criar códigos completos, corrigir arquivos e implementar melhorias.
- Priorizar execução e resultados, evitando análises repetitivas.
- Aproveitar funções, modelos e códigos existentes.
- Não recriar o projeto desnecessariamente.
- Explicar resumidamente as decisões técnicas importantes.
- Realizar testes disponíveis e informar resultados reais.
- Não declarar sucesso sem executar ou verificar.
- Manter código organizado, objetivo e com comentários naturais.
- Responder em português brasileiro.
- Ao explicar um plano ou resultado experimental, definir em linguagem simples
  o papel de cada etapa citada (piloto, preparo/congelamento, smoke test,
  treino, seleção interna, calibração, avaliação externa e retomada). Dizer
  o que cada etapa verifica ou produz, o que ainda não comprova e qual é o
  critério para avançar. Distinguir claramente "treino interno concluído"
  de "avaliação externa concluída"; não chamar piloto de treinamento.

## Treinamento
- Priorizar ajuste fino sobre treinamento do zero.
- Comparar os checkpoints existentes antes de escolher o inicial.
- Preservar modelos e resultados anteriores.
- Criar diretórios separados para novos experimentos.
- Registrar hiperparâmetros, métricas e checkpoints.
- Considerar limitações de GPU e armazenamento do Colab.
- Prevenir vazamento de dados entre treino e avaliação.
- Não usar as imagens externas para treinar, calibrar ou escolher
  hiperparâmetros, mesmo quando a meta externa ainda não for atingida.

## Dados de novos experimentos
- Esta regra vale somente para datasets novos, incorporados após a Fase 3.4;
  não alterar a coleta do OpenFake, os manifestos, os caminhos nem as
  execuções já preparadas das Fases 3.1–3.4 por causa dela.
- Antes de usar um novo dataset em treino, seleção, calibração ou teste,
  definir e registrar a versão, os arquivos/splits necessários, o tamanho
  estimado e a separação entre divisões. Baixar integralmente os arquivos
  escolhidos para o experimento; não acessar exemplos por streaming durante
  coleta, treino, validação ou avaliação. Não usar `streaming=True` como
  alternativa silenciosa quando o download não couber.
- Manter uma cópia persistente e um manifesto de integridade no Google Drive.
  Em cada sessão, conferir espaço livre no disco do Colab, copiar os dados
  necessários para `/content` e verificar contagem/hashes antes de iniciar
  qualquer processamento. Ler imagens do disco local durante a execução;
  salvar checkpoints, métricas e resultados no Drive. Após desconexão, a
  cópia local deve ser refeita a partir da cópia persistente.
- Se o dataset completo não couber no disco temporário ou na quota do Drive,
  interromper a preparação e propor uma seleção menor, versionada e
  justificável, antes de iniciar o experimento. Não misturar divisões nem
  incluir o conjunto externo independente nessa seleção de treino/calibração.
- Motivo: no OpenFake, o streaming serviu para selecionar e persistir imagens,
  mas a varredura remota foi lenta e dependente da conexão; acessar muitas
  imagens pequenas diretamente no Drive também causa latência e pode atingir
  limites de E/S. Dados locais antes da execução oferecem leituras repetidas
  mais previsíveis. A cópia no Drive preserva a retomada, pois `/content` é
  temporário. Detalhes em `docs/POLITICA_DADOS_FUTUROS.md`.

## Avaliação
- O conjunto externo não pode ser utilizado para treinamento,
  calibração ou escolha de hiperparâmetros.
- O benchmark externo RAISE/MNW v1 foi informado como congelado no Drive após
  reparar duplicatas e verificar 1.000 imagens reais + 1.000 sintéticas.
  As sete fases históricas foram avaliadas nele conforme saídas do usuário;
  a melhor acurácia foi 62,1% (Fase 3.4 v2). O conjunto já forneceu feedback
  e não deve servir para escolher ajustes da 3.5 nem ser apresentado como
  teste final independente de um modelo desenvolvido depois desse feedback.
- Fonte do benchmark v1: 1.000 fotografias RAISE e 1.000 imagens
  geradas da MNW, 25 de cada um de 40 grupos elegíveis, conforme seleção fixa em
  `experimentos/benchmark_2000.py`. O CSV oficial da RAISE exige formulário
  do pesquisador; a MNW é pública, mas o acesso real ao Git LFS deve passar
  por piloto antes da coleta completa. Não trocar fontes ou itens depois da
  primeira avaliação para melhorar métricas.
- A preparação do benchmark RAISE-1k encontrou seis pares RAW/JPEG reais
  exatamente duplicados antes de qualquer avaliação. O reparo com `RAISE_2k.csv`
  foi informado como concluído; `plan_repaired.json` preserva o plano original.
  Não refazer a seleção nem trocar itens após o congelamento.
- Diversidade de geradores é importante para medir generalização, mas não
  substitui o objetivo principal de melhorar a percepção do detector.
  Se muitos grupos com poucas imagens prejudicarem o diagnóstico, reduzir a
  diversidade somente em um benchmark futuro, definido antes da avaliação;
  nunca adaptar este conjunto ou os pesos/limiares às suas respostas.
- Baixar as imagens selecionadas das fontes oficiais uma vez, guardar a seleção
  imutável e o manifesto no Drive e, a cada nova sessão do Colab, copiar ambas
  as classes para `/content`, conferir contagens e hashes e avaliar a partir do
  disco local. Devido à quota gratuita, baixar cada RAW original temporariamente
  em `/content`, registrar seu hash e persistir somente o JPEG normalizado no
  Drive. Não baixar novamente os arquivos-fonte completos a cada teste;
  o arquivo único do Synthbuster tem 12,4 GB. Se a cópia local não couber,
  interromper a avaliação e revisar a fonte ou o tamanho antes de executá-la.
- Não usar `teste_externo/` para novas avaliações. Resultados anteriores com
  20 imagens permanecem históricos e não podem ser substituídos
  retrospectivamente por métricas de um conjunto ainda inexistente. Para
  comparar modelos antigos no novo conjunto, fazer novas avaliações com seus
  checkpoints e limiares congelados, em diretórios de resultado separados.
- Usar validação interna para selecionar modelos e limiares.
- Medir acurácia, AUC, acurácia balanceada, precisão,
  recall, F1 e especificidade.
- Investigar falsos positivos e falsos negativos.
- Comparar desempenho em geradores vistos e não vistos.
- Não prometer 85% sem evidência experimental.
- Antes da avaliação externa final, congelar checkpoint, temperatura e limiar.
  Registrar a execução sem ajustar o modelo a partir dessas imagens. Depois
  que as 20 imagens externas servirem de feedback para decidir novos trabalhos,
  elas não podem ser reutilizadas como teste independente final de outro modelo;
  reservar um conjunto novo para essa finalidade.

## Segurança
- Não apagar checkpoints ou notebooks originais.
- Não executar comandos Git destrutivos sem autorização.
- Não publicar datasets, modelos grandes ou credenciais.
- Não sobrescrever resultados de experimentos anteriores.
- Não inventar arquivos ou caminhos não verificados.

## Automação Colab e GitHub
- Branch de código para Colab: `fase3-ajuste-fino` do repositório privado `vicemnovo/TCC_Final`.
- O primeiro acesso usa `Notebooks/configuracao/configurar_github_colab.ipynb` no navegador do Colab. O segredo `GITHUB_TOKEN` deve ter apenas leitura de conteúdo desse repositório. A senha de desbloqueio não é salva no projeto.
- Nas sessões seguintes, `Notebooks/fase_3/rodar_fase3_4_colab.ipynb` solicita a senha e baixa um snapshot do commit para `/content/tcc_codigo/<sha>`. Não usar `files.upload()`, `userdata.get()` no runtime do VS Code, URL com token ou cabeçalhos persistidos no Git.
- A credencial criptografada fica em `/content/drive/MyDrive/TCC_Final/credenciais/github.enc`, ignorada pelo Git. Código sincronizado fica no runtime; datasets, checkpoints e resultados ficam no Drive. Não sobrescrever checkpoints ou resultados anteriores.
- O OpenFake pode ser baixado sem autenticação. Se for usado `HF_TOKEN`, configurá-lo uma vez no navegador com `Notebooks/configuracao/configurar_hf_colab.ipynb` e ler somente a credencial cifrada `credenciais/huggingface.enc` no runtime do VS Code. Não imprimir nem versionar o token.
- Alterações locais só chegam ao Colab depois de publicação autorizada na branch. Não fazer commit ou push sem autorização do usuário.
- `fase3_4_finetune.py auto` inicia ou retoma apenas quando explicitamente executado. O notebook não deve iniciar treinamento ao abrir.
- A v2 usa `experimentos/fase3_4_finetune_v2.py` e `Notebooks/fase_3/rodar_fase3_4_v2_colab.ipynb`. O treino só começa ao executar manualmente a célula 8 ou o modo `auto`; não reutilizar `modelos/fase3_4_finetune_v1/`.
- A Fase 3.5 usa `experimentos/fase3_5_dados.py`, `experimentos/fase3_5_finetune.py`
  e `Notebooks/fase_3/rodar_fase3_5_colab.ipynb`. O piloto baixa shards
  completos para `/content`, sem streaming; `prepare`, smoke e treino são
  ações manuais distintas. Preservar os manifestos e resultados anteriores.
- O piloto da 3.5 encontrou `real_source=n/a` para todas as fotos reais do
  shard 117. Não inventar fontes para essas imagens nem afirmar que a validação
  interna delas é separada por fonte. A divisão real é determinística por
  imagem e essa limitação deve constar do plano e das análises; geradores
  sintéticos continuam separados por grupo. Não usar RAISE/MNW para corrigir
  essa limitação ou selecionar hiperparâmetros.


## Tratamento de erros

Antes de executar ou publicar código que reutiliza notebooks, checkpoints ou
mensagens de fases anteriores, conferir estaticamente a ordem real das operações
e os logs correspondentes. Mensagens devem distinguir montagem da arquitetura,
carregamento dos pesos iniciais e retomada de checkpoint; não podem sugerir que
um checkpoint intermediário ainda é o modelo ativo. Corrigir mensagens herdadas
enganosas e, quando possível, adicionar um teste de regressão antes do smoke
test ou do treinamento.

Quando um erro de execução for apresentado:

- Examinar o traceback e os arquivos envolvidos.
- Identificar a causa antes de modificar o código.
- Corrigir diretamente os arquivos do projeto.
- Verificar dependências entre funções, módulos e células.
- Executar testes disponíveis para validar a correção.
- Não modificar configurações de treinamento sem necessidade.
- Não sobrescrever checkpoints ou resultados anteriores.
- Evitar repetir verificações já realizadas.
- Não solicitar que o usuário faça alterações manuais
  quando for possível realizá-las diretamente no repositório.
- Considerar as diferenças entre o VS Code local e o
  runtime remoto do Google Colab.
- Informar o que foi alterado e o que precisa ser
  executado novamente.

## Progresso de execuções longas

- Células que baixam dados, preparam amostras, fazem verificações extensas ou treinam devem mostrar logs em tempo real e um aviso periódico durante intervalos sem saída.
- O aviso periódico indica que o processo continua ativo, mas não substitui métricas, checkpoints ou confirmação de avanço efetivo.
- Manter a execução do treinamento como ação manual no notebook.
