# Novo benchmark externo — Mirage-Test 2000 v1

## Finalidade e fonte

Este é um **novo teste**, separado dos 20 casos históricos e do RAISE/MNW.
O código prepara 2.000 imagens do conjunto público
[Mirage-Test](https://huggingface.co/datasets/Yunncheng/Mirage-Test): 1.000
rotuladas reais e 1.000 sintéticas. O [artigo dos autores](https://arxiv.org/abs/2511.08423)
apresenta o dataset como teste de detecção em diferentes conteúdos. O cartão
informa fotografias reais de plataformas como Pexels e imagens sintéticas
geradas por modelos como Stable Diffusion e FLUX. São declarações dos autores;
hashes comprovam identidade de arquivos, **não** a origem factual de cada foto.
O repositório declara licença MIT, mas a origem de terceiros pode impor
condições adicionais. Manter imagens no Drive privado e publicar apenas
resultados agregados.

O repositório inteiro tem aproximadamente **41,2 GB e 49.000 itens**. O script
baixa integralmente apenas o índice `test.parquet` e os **2.000 arquivos
escolhidos**, sem streaming. O tamanho dos selecionados não é conhecido antes
do piloto; 2.000 vezes o tamanho médio do repositório sugere cerca de 1,7 GB,
sem garantia. O preparo exige pelo menos 4 GB livres em `/content` e 3 GB no
Drive; interrompe se não houver espaço suficiente.

## Seleção prévia e independência

- O índice é fixado por SHA do commit do dataset e por hash SHA-256 do
  `test.parquet`; 250 reais e 250 sintéticas são selecionadas em cada um dos
  domínios Animal, Human, Object e Scene. Anime é excluído porque o cartão
  informa apenas sintéticas nesse domínio.
- A seleção usa uma semente fixa e nomes dos arquivos. Para sintéticas,
  distribui as vagas entre os grupos de geradores declarados no caminho.
  Nenhuma predição influencia essa escolha.
- Antes de congelar o manifesto, o preparo confere duplicatas exatas dentro
  do novo conjunto e sobreposição exata com OpenFake treino/teste,
  CommunityForensics-Small selecionado, RAISE/MNW e cópias locais das 20
  imagens históricas, quando presentes. Se houver conflito, interrompe sem
  substituir imagens depois de ver previsões.
- A ausência de hash repetido **não prova** que duas versões editadas da mesma
  fotografia tenham origens distintas. Há possível fonte Pexels tanto no novo
  dataset quanto no OpenFake; registrar essa limitação na interpretação.
- O plano, os arquivos e os hashes ficam em
  `/content/drive/MyDrive/TCC_Final/dados_externos/benchmark_externo_mirage_2000_v1/`.
  Uma pasta `.incompleto` permite retomar o mesmo plano após desconexão.
  A cópia de avaliação fica em `/content/benchmark_externo_mirage_2000_v1/`.

## Modelo e protocolo congelados

Somente a **Fase 3.5** é avaliada nesta versão. O checkpoint final registrado
tem SHA-256 `1dd78d4cd32106d1471d388fa3a96a094c3906e7860ed4ad5ec3d7acad9b9045`.
O script confere o hash e lê do próprio checkpoint a temperatura e o limiar,
comparando-os às métricas internas (aproximadamente **0,440947** e **0,445**).
Usa cinco recortes de 224 px e média dos logits, como no diagnóstico interno.
Uma imagem com lado curto menor que 224 px é ampliada uma vez por interpolação
bicúbica antes dos recortes; a contagem desses casos é registrada. Essa regra
está definida **antes** de qualquer predição externa.

O teste não escolhe checkpoint, temperatura, limiar, recortes nem
hiperparâmetros. Relata acurácia com intervalo de Wilson de 95%, AUC,
acurácia balanceada, precisão, recall sintético, especificidade real, F1,
matriz de confusão e acerto por domínio/classe. As probabilidades por imagem
ficam em resultado privado separado:
`/content/drive/MyDrive/TCC_Final/resultados/benchmark_externo_mirage_2000_v1/fase3_5_cfsmall_v1/`.
Resultados antigos e checkpoints não são sobrescritos. Depois de observadas,
essas 2.000 imagens não poderão ser o teste final independente de uma variante
desenvolvida em resposta a seus erros.

## Execução manual no Colab via VS Code

Abra `Notebooks/fase_3/avaliar_fase3_5_mirage_2000_colab.ipynb` e conecte a
GPU do Colab. A branch remota precisa conter o novo script e notebook; as
alterações locais só chegam ao runtime após publicação autorizada.

1. Células **1–6**: leia a introdução, verifique GPU e dependências, monte o
   Drive, desbloqueie o GitHub, sincronize o snapshot e configure logs. Se a
   credencial Hugging Face cifrada já existir no Drive, a célula 6 solicita
   sua senha; caso contrário o dataset público é acessado sem token.
2. Célula **8** (`pilot`): lê somente o índice e mostra commit, contagens e
   seleção prevista. Não baixa as imagens nem cria plano no Drive.
3. Célula **9** (`prepare`): cria o plano antes dos downloads, baixa as 2.000
   imagens para `/content`, copia para o Drive, confere hashes e congela o
   manifesto. Pode demorar e mostra progresso; execute novamente se cair.
4. Célula **10** (`stage`): copia o conjunto congelado do Drive para `/content`
   e verifica cada hash. Repita em cada sessão nova; não baixa novamente da
   fonte.
5. Célula **11**: confira somente leitura o manifesto, commit do dataset,
   hash do checkpoint, temperatura, limiar e a presença da cópia local.
6. Célula **12** (`evaluate`): **inicia manualmente a avaliação única**. O
   script confere integridade, carrega o checkpoint e faz 2.000 inferências.
   A célula **13** lê o resultado salvo sem reavaliar.

Nenhuma célula inicia treinamento. Se o piloto mostrar estrutura ou contagens
incompatíveis com o código, **não execute `prepare`**: registre a saída para
corrigir o leitor do índice sem alterar os critérios de seleção com base em
predições. A avaliação remota, a coleta e o tamanho real ainda não foram
executados nesta sessão local.
