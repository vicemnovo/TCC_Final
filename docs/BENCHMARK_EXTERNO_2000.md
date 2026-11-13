# Benchmark externo novo: 1.000 RAISE + 1.000 MNW

Este protocolo **não altera** as avaliações históricas. As 20 imagens externas
já testadas continuam identificadas como teste antigo, com origem incompleta.
O piloto, a coleta e o reparo de seis duplicatas reais foram executados no
Colab. A saída informou **benchmark congelado**, ausência de sobreposição exata
com OpenFake v2 e com as 20 imagens antigas, e cópia local verificada de 1.000
reais + 1.000 sintéticas. As sete fases históricas foram avaliadas nesse
conjunto conforme saídas do Colab fornecidas pelo usuário; a melhor acurácia
foi **62,1% na Fase 3.4 v2**. O conjunto já forneceu feedback e não poderá
servir como teste final independente de versões desenvolvidas depois dele.

## Origem, acesso e limite de uso

- **Reais:** [RAISE](https://loki.disi.unitn.it/RAISE/download.html), 1.000
  fotografias RAW nativas de câmera. O site exige formulário com nome,
  afiliação e e-mail e aceite dos termos de pesquisa/ensino. O [guia
  oficial](https://loki.disi.unitn.it/RAISE/guide.html) indica que o CSV
  entregue contém URLs individuais na coluna `NEF`. Não foi identificada
  exigência de token de API. O usuário precisa obter o CSV e colocá-lo em
  `/content/drive/MyDrive/TCC_Final/entradas/RAISE_1k.csv`. Não salvar dados
  pessoais ou URLs assinadas no Git.
  Embora o CSV tenha 1.000 URLs NEF diferentes, a auditoria dos bytes baixados
  mostrou apenas 994 conteúdos reais únicos. Para manter 1.000 fotos únicas,
  obter também o CSV de metadados RAISE-2k, não o pacote de 2.000 imagens, e
  colocá-lo em `TCC_Final/entradas/RAISE_2k.csv`.
- **Sintéticas:** [MNW, Microsoft–Northwestern–WITNESS](https://github.com/microsoft/MNW),
  1.000 arquivos de `AI_Images/` selecionados em 40 pastas de geradores,
  25 por pasta. A fonte é pública e não exige conta/token para navegar.
  Imagens individuais usam Git LFS; o piloto do notebook deve confirmar que
  os bytes completos podem ser baixados sem autenticação ou limite impeditivo.
  O [README da MNW](https://github.com/microsoft/MNW#readme) autoriza uso para
  avaliação, não treinamento nem uso comercial.

O token GitHub **já existente** no projeto é usado somente para ler o código
privado do TCC. Não é passado ao servidor da MNW. Se a RAISE ou a MNW mudar
as condições de acesso, interromper e consultar os termos; não inserir
credenciais em código ou logs.

## Seleção, integridade e custo

`experimentos/benchmark_2000.py` usa uma semente fixa para escolher as 1.000 URLs
RAISE e 25 arquivos em cada um de 40 grupos MNW. Os grupos elegíveis são escolhidos
deterministicamente entre aqueles com pelo menos 25 arquivos. Ele fixa o commit MNW
e salva `plan.json` antes de baixar. Se houver desconexão, reutiliza o plano
existente e os arquivos já registrados; não escolhe outras imagens. A pasta
`dados_externos/benchmark_externo_raise_mnw_2000_v1.incompleto/` só é
renomeada para `benchmark_externo_raise_mnw_2000_v1/` após:

1. 2.000 JPEGs derivados estarem presentes e legíveis;
2. SHA-256 individual das fontes e dos JPEGs registrado e ausência de duplicatas exatas internas;
3. comparação de hashes com as 3.979 imagens das partições OpenFake v2 e
   com as 20 imagens externas históricas.

### Reparo dos seis pares RAISE duplicados

O diagnóstico inicial do usuário encontrou 4.000 hashes registrados, 3.988 distintos:
seis pares de RAWs Nikon D90 e seus seis pares de JPEGs derivados são idênticos.
Não houve duplicatas MNW reportadas nem hashes faltantes. O reparo posterior
substituiu as seis fotografias repetidas antes de congelar o benchmark.

Após obter `RAISE_2k.csv` da [fonte oficial](https://loki.disi.unitn.it/RAISE/download.html),
executar a célula **7B** do notebook. O modo `repair` lê o plano original e os
hashes, escolhe candidatas externas ao RAISE-1k por ordem determinística e
baixa somente as candidatas necessárias para confirmar que RAW e JPEG são
distintos dos 1.994 itens preservados. Ele grava `plan_repaired.json` com hash
do plano original, hash do CSV adicional, IDs retirados e substituições. Não
sobrescreve `plan.json`, não apaga os JPEGs baixados e não usa predições do modelo.
Depois executar novamente a célula **7**: ela reutiliza as 1.994 imagens ativas
e baixa apenas as seis substitutas antes de concluir a auditoria. Os arquivos
duplicados antigos podem continuar na pasta, mas ficam fora do manifesto final.
Esse procedimento já foi informado como concluído; não repeti-lo para a avaliação.

As fotos RAW são convertidas para JPEG RGB qualidade 90 e limite de 2.048 px;
as MNW passam pelo mesmo processamento. Os originais são baixados um a um para
`/content` e descartados após o registro de seus hashes; somente os JPEGs
derivados são persistidos no Drive. Isso evita guardar 10–22 GB ou mais de RAW
na quota gratuita. O CSV de origem e o commit MNW fixam a procedência. Isso reduz
atalhos triviais de extensão, mas **não prova** equivalência de compressão,
cena ou origem semântica. Revisar amostras e possíveis quase-duplicatas antes
de fazer alegações amplas de independência.

Estimativa inicial: a RAISE publica exemplos de arquivos RAW de cerca de
10–22 MB; 1.000 exigem cerca de 10–22 GB de tráfego no preparo, mas não de
armazenamento persistente. O espaço dos JPEGs deve ser conferido no Colab e
no Drive; nenhum tamanho final foi medido.
Não baixar o acervo inteiro nem usar streaming em inferência. Cada sessão
copia os 2.000 JPEGs congelados do Drive para `/content` e valida hashes.

## Notebook e resultados

Abra `Notebooks/fase_3/comparativos/avaliar_fases_benchmark_2000_colab.ipynb`. As primeiras
células montam o Drive, desbloqueiam a credencial GitHub existente e sincronizam
o código publicado na branch `main`. Depois execute, na ordem:

1. célula intitulada **"5. Dependências e entradas"**: presença do CSV RAISE;
2. **"6. PILOTO MANUAL"**: uma imagem de cada fonte;
3. **"7B. Reparo MANUAL"**: após o CSV RAISE-2k e publicação do código,
   reexecutar as células 3, 4 e 5 para sincronizar o snapshot novo, então criar
   o plano derivado preservando o original;
4. **"7. PREPARAÇÃO MANUAL"** novamente: completar apenas as substitutas;
5. **"8. Cópia/verificação"**: cópia íntegra para `/content` em cada sessão;
6. **"9. Conferência dos checkpoints"**: conferir modelos e executar
   manualmente cada avaliação desejada nas células seguintes.

`experimentos/avaliar_benchmark_2000.py` lê os checkpoints congelados das
Fases 1, 2, 3.1, 3.2, 3.3, 3.4 v1 e v2. Cada execução grava métricas e
predições em um diretório separado sob
`resultados/benchmark_externo_raise_mnw_2000_v1/` e se recusa a sobrescrever
um resultado existente. Fase 1 não tem temperatura persistida no checkpoint:
o novo protocolo usa T=1 e limiar 0,5, sem afirmar que isso reproduz uma
calibração histórica. Fase 2 usa resize 224; Fases 3 usam cinco recortes.

A primeira tentativa de avaliar a Fase 1, no snapshot `30cf718`, falhou antes
da inferência com `KeyError: 'Phase1Detector'`: a classe estava sendo buscada
no notebook da Fase 3.3. A correção carregou a classe da Fase 1 do notebook
original e foi validada no Colab com 2.000 inferências concluídas. Os sete
resultados foram gravados em diretórios separados, com manifesto e script de
avaliação iguais. Não recalibrar os checkpoints com estes resultados.

Comparar as fases **descritivamente** nas mesmas 2.000 imagens. Não usar
predições externas para escolher arquitetura, pesos, limiar, hiperparâmetros
ou itens do teste. Melhorias futuras devem ser sustentadas por treino e
validação internos; um novo teste independente exigiria outro conjunto
predefinido. O benchmark mede percepção em fontes específicas, não garante
acurácia de 85% em toda a internet.
