
# TCC_Final — Instruções para o Codex

## Objetivo
Concluir o projeto de detecção robusta de imagens reais e imagens
sintéticas geradas ou alteradas por inteligência artificial, seguido
pelo desenvolvimento de uma aplicação web com API.

A detecção de vídeos está fora do escopo atual por prazo e custo;
poderá ser considerada em trabalho futuro, sem direcionar os
experimentos ou entregas desta etapa.

Meta: alcançar pelo menos 85% de acurácia em um conjunto externo
independente, com imagens não utilizadas no treinamento.

A meta é um objetivo experimental, não um resultado já alcançado.

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
- `Notebooks/fase_3/`: Fases 3.1, 3.2, 3.3 e interface Colab da 3.4.
- `Notebooks/configuracao/`: configuração inicial de credenciais no Colab.
- `experimentos/`: código Python principal dos experimentos; a Fase 3.4 depende
  de `Notebooks/fase_3/fase3.3_compressao.ipynb` no snapshot do GitHub.
- `docs/`: instruções de execução; `README.md` é o índice do projeto.
- `modelos/`, `minhas_imagens/` e `validacao_fase3/` contêm artefatos
  legados. A pasta local `teste_externo/` foi retirada; resultados históricos
  permanecem documentados e cópias das imagens existem nos outros caminhos.
- Não mover dados, checkpoints ou resultados para acompanhar mudanças de código.

## Prioridade atual
Continuar a Fase 3, especialmente o ajuste fino do detector
para melhorar a generalização em imagens externas.

Antes de novos treinos, auditar distribuição, formatos, resoluções e
duplicatas dos dados; avaliar generalização por gerador e fonte;
planejar mudanças incrementais de robustez. Depois de estabilizar
e avaliar o detector de imagens, desenvolver a aplicação web e a API.

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
- Para novas avaliações externas, preparar uma versão congelada e documentada
  com alvo de 200 imagens reais e 200 sintéticas, procedência e hashes por
  arquivo. Esse conjunto ainda não foi obtido; não declarar avaliações nele
  como executadas. O tamanho final depende da disponibilidade e da auditoria.
- Baixar as imagens selecionadas das fontes oficiais uma vez, guardar a seleção
  imutável e o manifesto no Drive e, a cada nova sessão do Colab, copiar ambas
  as classes para `/content`, conferir contagens e hashes e avaliar a partir do
  disco local. Não baixar novamente os arquivos-fonte completos a cada teste;
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
