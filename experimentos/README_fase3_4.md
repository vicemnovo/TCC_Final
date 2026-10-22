# Execução da Fase 3.4

## Primeira vez

1. Publique, após revisão e autorização, os arquivos deste experimento na branch privada `fase3-ajuste-fino`. Alterações apenas no VS Code não ficam disponíveis para o runtime remoto.
2. No navegador do Colab, abra `experimentos/configurar_github_colab.ipynb`. Configure no painel de Segredos um `GITHUB_TOKEN` fine-grained limitado a `vicemnovo/TCC_Final`, permissão **Contents: Read-only**, e autorize o notebook a lê-lo. Execute as duas células; escolha uma senha forte e guarde-a fora do projeto. O token cifrado será criado automaticamente em `TCC_Final/credenciais/github.enc` no Drive.
3. Confirme que os checkpoints das Fases 2 e 3.3 estão em `/content/drive/MyDrive/TCC_Final/modelos/`.

## Cada sessão pelo VS Code

Abra `experimentos/rodar_fase3_4_colab.ipynb`, conecte a GPU do Colab e execute as células 1 a 5. Informe a senha quando solicitada. A célula 4 baixa um snapshot imutável da branch para `/content/tcc_codigo/<sha>`; não modifica o Drive nem arquivos locais do VS Code. A célula 5 verifica checkpoint e dependências.

Na primeira sessão, execute a célula 6 para coletar o OpenFake em `TCC_Final/dados_fase3_4/` no Drive. Depois execute a 7 para testar dados, arquitetura, checkpoint, forward e gravação sem treino. Em sessões posteriores, dados e smoke existentes são preservados.

As células de preparação, smoke test e treino mostram a saída do script conforme ela é produzida. Se não houver novas linhas, exibem um aviso a cada 30 segundos com o tempo decorrido; esse aviso confirma que o processo ainda está ativo, mas não garante que houve avanço nos dados.

Quando quiser treinar, retire `#` da linha de comando da célula 8 e execute-a. O modo `auto` começa se a pasta de execução não existir, retoma de `last.pt` após desconexão e encerra sem alterações se `final_metrics.json` já existir. O treino salva em `TCC_Final/modelos/fase3_4_finetune_v1/`; checkpoints, histórico e métricas das experiências anteriores não são alterados.

O conjunto externo de 20 imagens é consultado somente por hashes para bloquear duplicação em treino ou validação. A avaliação final usa o teste OpenFake depois de fixar época, temperatura e limiar pelas duas validações internas. Uma avaliação externa independente posterior deve ser feita uma única vez com o modelo congelado.

## Limite importante

A branch remota precisa conter a versão desejada de `fase3_4_finetune.py` e do notebook original `fase3.3_compressao.ipynb`. A autenticação cifrada não transfere alterações locais não publicadas.
