# Espelho de notebooks e scripts no Google Drive

## Onde ficam os arquivos

O GitHub continua sendo a fonte versionada do código. Em cada sessão Colab,
um commit da branch `main` é baixado para
`/content/tcc_codigo/<sha>` para execução. O notebook
[`sincronizar_codigo_drive_colab.ipynb`](../Notebooks/configuracao/sincronizar_codigo_drive_colab.ipynb)
faz também uma cópia persistente dos arquivos `.ipynb` e `.py` desse **mesmo
commit** para `/content/drive/MyDrive/TCC_Final`, mantendo as pastas:

| Repositório | Espelho no Drive |
| --- | --- |
| `Notebooks/fase_3/fase3_9/rodar_fase3_9_tgif_colab.ipynb` | `TCC_Final/Notebooks/fase_3/fase3_9/rodar_fase3_9_tgif_colab.ipynb` |
| `Notebooks/fase_3/fase3_9/fase3_9_tgif_finetune.py` | `TCC_Final/Notebooks/fase_3/fase3_9/fase3_9_tgif_finetune.py` |
| `tests/test_espelhar_codigo_drive.py` | `TCC_Final/tests/test_espelhar_codigo_drive.py` |

O espelho cobre **todos** os `.ipynb` e `.py` do snapshot, inclusive fases
anteriores. Não copia arquivos locais ainda sem commit e push. Não copia
datasets, pesos, resultados, credenciais nem outros tipos de arquivo.

## Primeira sincronização e atualizações

1. No VS Code, publique na branch `main` as alterações de código
   que devem chegar ao Drive. Mudanças apenas locais não aparecem no snapshot.
2. Abra `Notebooks/configuracao/sincronizar_codigo_drive_colab.ipynb` com o
   runtime Colab da conta correta. GPU não é necessária.
3. Execute as células **1–5 em ordem**. A #2 monta o Drive; a #3 pede a senha
   da credencial GitHub cifrada que já existe no Drive; a #4 baixa o commit;
   a #5 copia e verifica os arquivos. Não há upload manual via VS Code nem
   criação manual de diretórios.
4. Confira o número de arquivos, o commit e o manifesto em
   `TCC_Final/codigo_sync/<commit>.json`. Os caminhos mostrados ao final da
   célula #5 devem estar presentes no Drive. Para uma nova versão de código,
   publique primeiro o commit e repita as células 1–5.

Se a sessão cair no meio da cópia, execute novamente a #5 após restaurar
as células anteriores no runtime. Arquivos já iguais são pulados; um manifesto
só é gravado após todos os arquivos serem conferidos. Um arquivo do Drive
com conteúdo diferente é copiado antes para
`TCC_Final/codigo_historico/<sha256-do-conteudo-antigo>/<caminho-original>`;
depois o caminho principal recebe a versão do GitHub. O programa nunca apaga
essas cópias anteriores. O caminho de checkpoint e os dados não são alterados.

O espelho facilita abrir notebooks e scripts pelo Drive, mas os experimentos
continuam executando o snapshot de commit em `/content/tcc_codigo/<sha>`.
Isso evita misturar código de commits diferentes durante um treino. Não
edite a cópia no Drive esperando que ela atualize o GitHub automaticamente:
edite no repositório local e publique a branch.
