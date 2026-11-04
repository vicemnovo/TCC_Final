# Guia básico de Git e Colab para o grupo

Este repositório guarda **código, notebooks e documentação**. Novas saídas de
execução dos notebooks devem ser revisadas antes de entrar em um commit. O
Google Drive guarda datasets, manifestos de execução, checkpoints e
resultados. O GitHub não substitui o Drive; `/content` do Colab é temporário.
Trabalhamos na branch `fase3-ajuste-fino`.

## Termos usados aqui

| Termo | Significado prático |
| --- | --- |
| *Commit* | Registro local de uma mudança com mensagem e identificador |
| *Push* | Envio dos commits locais ao GitHub |
| *Branch* | Linha de desenvolvimento; a atual é `fase3-ajuste-fino` |
| *Snapshot* | Cópia de um commit específico baixada pelo notebook no Colab |
| *Checkpoint* | Pesos e estado do modelo salvos no Drive, fora do Git |

## Antes de trabalhar em outro computador

No PowerShell, dentro de `E:\TCC\TCC_Final`:

```powershell
git status -sb
git branch --show-current
git fetch origin
```

Confira que a branch é `fase3-ajuste-fino`. Se `git status -sb` mostrar
arquivos modificados, revise-os antes de atualizar: podem ser saídas de
notebook ou trabalho de outro integrante. Com o trabalho local identificado,
atualize a branch:

```powershell
git pull --ff-only origin fase3-ajuste-fino
```

Se o `pull` recusar por mudanças locais ou commits divergentes, pare e revise
com o grupo. Não use `reset --hard`, `clean -fd` ou `push --force` para resolver.

## Fazer um commit pequeno e compreensível

1. Execute `git status --short` e `git diff -- nome/do/arquivo`.
2. Adicione **somente** os arquivos pretendidos, por exemplo:

   ```powershell
   git add -- README.md ESTADO_PROJETO.md docs/GUIA_GIT_E_COLAB.md
   ```

3. Revise exatamente o que será publicado:

   ```powershell
   git diff --cached --check
   git diff --cached --stat
   git diff --cached
   ```

4. Crie uma mensagem que diga *o que mudou e por quê*. Use um assunto curto
   e, para várias alterações, um corpo com os arquivos ou temas principais:

   ```powershell
   git commit -m "docs: atualizar retomada da Fase 3.8" -m "Registra preparo e smoke verificados; explica células pendentes e limites do Mirage."
   ```

5. Publique e confira o resultado:

   ```powershell
   git push origin fase3-ajuste-fino
   git status -sb
   git log -1 --oneline
   ```

O `status -sb` deve mostrar a branch acompanhando a remota. Se ainda houver
arquivos modificados, confira quais são; o push pode ter funcionado mesmo
com saídas locais de notebooks não incluídas no commit. Não use `git add .`
sem revisar, pois ele também tenta incluir arquivos novos indesejados.

## Notebooks e Colab

- Salve as saídas necessárias no Drive ou em uma cópia local. Em geral, limpe
  saídas e contadores de execução antes de publicar um notebook. Uma saída
  curta que documente uma etapa executada pode ser versionada após revisão
  explícita de credenciais, dados pessoais, imagens e tamanho do arquivo.
  Saídas do Colab não são mudanças no código, mesmo quando entram no Git.
- Alterar um arquivo no VS Code não muda o snapshot já carregado no Colab.
  Depois do push, execute novamente as células de desbloqueio GitHub,
  sincronização do snapshot e definição dos comandos do notebook. Confira o
  SHA impresso antes de preparar dados ou treinar.
- O notebook da 3.8 é
  [`rodar_fase3_8_colab.ipynb`](../Notebooks/fase_3/rodar_fase3_8_colab.ipynb).
  O roteiro de execução está em [`README_fase3_8_execucao.md`](README_fase3_8_execucao.md).
- Após desconexão, monte o Drive e copie dados necessários para `/content`.
  `last.pt` permite retomar treino incompleto. Não repita preparo ou smoke
  concluídos apenas para recuperar variáveis.

## O que fica fora do Git

Não adicione tokens, senhas, arquivos `.enc`, datasets, imagens de avaliação,
checkpoints `.pt`, logs de treino ou resultados grandes. A pasta
`/content/drive/MyDrive/TCC_Final/credenciais/` não deve ser sincronizada com
o GitHub. O `.gitignore` protege **novos** arquivos que combinam com suas
regras; ele não remove arquivos já versionados. Há imagens e CSVs históricos
sob `minhas_imagens/` e `validacao_fase3/` que já estavam no índice: preservar
esse histórico até uma decisão explícita do grupo sobre sua publicação.

Antes de cada push, execute `git diff --cached --name-only` e procure por
credenciais, imagens, modelos e saídas de notebooks. Em caso de dúvida,
interrompa o push e revise o commit local.
