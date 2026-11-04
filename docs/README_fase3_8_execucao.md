# Fase 3.8 — AIGI pareado

**Estado em 03/11/2026:** preparo, cópia local e smoke test concluídos conforme
saídas salvas no notebook. A célula `# 10. TREINO MANUAL` ainda não tem saída
local registrada. Em sessão já preparada, ela é a próxima ação; em sessão nova,
execute `# 1` a `# 5`, depois `# 8` e `# 10`. Não repita o preparo concluído.

## Para que serve

A 3.7 acrescentou exemplos, mas a auditoria dos novos dados mostrou uma
associação entre classe, formato e resolução: reais JPEG com lado curto mediano
de 428 px e sintéticas PNG com lado curto de 512 px. Isso pode facilitar um
atalho de classificação; a auditoria não provou que o modelo o usa. A 3.8
adiciona pares de imagens reais e sintéticas vinculados pelo mesmo `uid`, com
pré-processamento JPEG e lado curto de 256 px para as duas classes. A
arquitetura ViT + FFT + EfficientNet-B0, o checkpoint inicial da 3.3, as
transformações, o amostrador e o limite de 12 épocas permanecem iguais aos da
3.7. Imagens de usuário continuam sendo classificadas só pelos pixels.

O [AIGI](https://huggingface.co/datasets/Coxy7/AIGI-Detection-Quality-Paradox)
fornece as sintéticas e metadados das fotografias reais, vinculados por UID.
As fotografias precisam ser baixadas de suas fontes. O piloto no Colab fixou
o commit `9244882a51dbe33e658fe514488692155d20e5dd`. O shard inicial tinha
1.600 FLUX.1-dev; das 16 fotos reais tentadas, 15 abriram e só duas tinham
lado curto de pelo menos 512 px. Por isso o preparo exige 256 px, suficiente
para cinco recortes de 224 sem ampliação. Esse redimensionamento pode perder
artefatos finos e não torna as distribuições antigas idênticas. Rótulos e
licenças das fontes reais ainda precisam de conferência antes de divulgação
das imagens; o resultado de 16 URLs não garante disponibilidade das demais.

## Plano fixado antes do treino

- Seis shards Parquet iniciais, índices 0, 3, 5, 8, 10 e 13, mais o CSV real.
  Como a primeira tentativa terminou com apenas 75/80 pares `SDXL_1.0`,
  um complemento imutável acrescenta os shards completos 9, 11 e 14 para
  `SDXL_1.0`, `SD_2.1` e `SD_3`. São nove shards no total, cerca de 4,5 GB
  de downloads em `/content`, sem streaming de exemplos.
- Até 80 pares completos de cada gerador. Quatro geradores entram no treino;
  `PA_alpha` fica só na seleção de checkpoint e `SD_2.1` só na calibração.
  O programa exige 80 em cada grupo e interrompe sem treinar se faltarem
  pares. São 480 pares, 960 imagens, **somente se a preparação concluir**.
- Cada UID fica em uma única divisão. O manifesto registra origem, formato,
  tamanho e SHA-256 original/processado de cada arquivo. Há verificação de
  sobreposição exata com dados anteriores e externos reservados; ausência de
  hash igual não prova independência visual.
- Seleção e estado incompleto persistem em
  `/content/drive/MyDrive/TCC_Final/dados_fase3_8/aigi_pares_256_v1.incompleto/`;
  `plan.json` e `state.json` da primeira tentativa são preservados, e
  `plan_extension_1.json` registra os três shards adicionais antes da
  seleção dos novos pares. Os 315 pares reais já aceitos são reaproveitados.
  Após concluir, os dados ficam em `dados_fase3_8/aigi_pares_256_v1/`.
  A primeira materialização encontrou uma sintética menor que 256 px apesar
  dos metadados. O preparo agora verifica os bytes de cada sintética antes de
  congelar o manifesto; registra rejeições em `state.json` e procura outro par
  do mesmo gerador, sem ampliar imagens pequenas. Arquivos incompletos já
  gravados não são apagados ou incluídos no manifesto final.
  Checkpoints e métricas novos ficam em `modelos/fase3_8_aigi_pares256_v1/`.
  Não sobrescrever diretórios anteriores.

## Execução no Colab pelo VS Code

Abra `Notebooks/fase_3/rodar_fase3_8_colab.ipynb` no commit publicado da branch
`fase3-ajuste-fino`. Conecte GPU e execute as células de código **1 a 5** em
ordem: dependências, Drive, desbloqueio da credencial GitHub, snapshot do
repositório e função de progresso. A célula **6** apenas registra que o piloto
já foi analisado. A numeração abaixo refere-se aos comentários `# N.` dentro
das células, não à posição incluindo Markdown.

1. Execute **`# 7. PREPARO MANUAL`**. Baixa os nove shards completos, testa
   fotos reais, fixa os pares e produz o manifesto no Drive. Pode levar tempo;
   os logs mostram progresso e o notebook indica que o processo continua
   ativo. Se desconectar, execute novamente; a seleção já registrada é
   retomada. Pare e envie o erro se faltar quota, espaço, pares ou licença.
   Quando `credenciais/huggingface.enc` existe no Drive, a célula pede a senha
   de desbloqueio e passa o `HF_TOKEN` somente ao processo de preparo, sem
   mostrá-lo. O Secret do navegador não chega automaticamente ao kernel do
   VS Code. Se o arquivo cifrado ainda não existir, use uma vez
   `Notebooks/configuracao/configurar_hf_colab.ipynb` no navegador do Colab.
   O AIGI é público e permite preparo sem autenticação, sujeito aos limites
   de acesso anônimo.
2. Execute **`# 8. COPIA LOCAL`** em toda sessão nova. Copia e confere por hash
   os dados 3.5, 3.7 e 3.8 de Drive para `/content`.
3. Execute **`# 9. SMOKE MANUAL`**. Confere divisões, hashes, GPU, arquitetura,
   checkpoint da 3.3, passagem forward e gravação. Não treina. Avance só com
   `Smoke OK` e contagens coerentes.
4. Execute **`# 10. TREINO MANUAL`** para iniciar ou retomar automaticamente
   de `last.pt`. Confira épocas, AUC de seleção, perda, modelo escolhido e
   calibração. A célula **`# 11. LEITURA INTERNA`** mostra os artefatos.
5. Depois de `training_complete.json`, execute **`# 12. MIRAGE`**,
   **`# 13. CONGELAMENTO`**, **`# 14. AVALIACAO EXTERNA`** e
   **`# 15. LEITURA`**. Temperatura e limiar vêm apenas da calibração interna.

O Mirage de 2.000 imagens já foi observado. Seu uso nesta fase é comparação
descritiva com checkpoints e limiares congelados, nunca fonte para treinar,
calibrar ou escolher hiperparâmetros. A 3.7 obteve 62,3% nele conforme saída
do usuário; a 3.8 ainda não foi treinada nem avaliada. Para demonstrar 85% em
teste final independente será necessário um conjunto novo, não observado.

Se `# 10` informar que `run_live` ou outra variável está ausente, o kernel
perdeu o estado ou as células iniciais não foram executadas nesta sessão.
Execute novamente `# 1` a `# 5`, depois `# 8` para restaurar os dados locais,
e só então `# 10`. Esse erro acontece antes de o treino começar; não repita
o preparo dos pares da célula `# 7`.
