
# TCC_Final — Instruções para o Codex

## Objetivo
Concluir o projeto de detecção de imagens sintéticas geradas ou
alteradas por inteligência artificial.

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

## Prioridade atual
Continuar a Fase 3, especialmente o ajuste fino do detector
para melhorar a generalização em imagens externas.

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

## Avaliação
- O conjunto externo não pode ser utilizado para treinamento,
  calibração ou escolha de hiperparâmetros.
- Usar validação interna para selecionar modelos e limiares.
- Medir acurácia, AUC, acurácia balanceada, precisão,
  recall, F1 e especificidade.
- Investigar falsos positivos e falsos negativos.
- Comparar desempenho em geradores vistos e não vistos.
- Não prometer 85% sem evidência experimental.

## Segurança
- Não apagar checkpoints ou notebooks originais.
- Não executar comandos Git destrutivos sem autorização.
- Não publicar datasets, modelos grandes ou credenciais.
- Não sobrescrever resultados de experimentos anteriores.
- Não inventar arquivos ou caminhos não verificados.
