# Política de dados para experimentos futuros

Esta política se aplica **apenas a datasets novos**, introduzidos depois dos
experimentos da Fase 3.4. Não modifica o OpenFake já preparado, seus manifestos,
as divisões registradas nem o treino v2 em andamento.

## Regra operacional

1. Antes da coleta, identificar a versão do dataset, os arquivos e as divisões
   necessários, estimar tamanho e conferir espaço disponível no Drive e no
   disco temporário do Colab. Registrar licenças e proveniência quando houver.
2. Baixar **por completo os arquivos selecionados para aquele experimento**,
   sem `streaming=True` ou leitura remota de exemplos durante treino,
   validação, calibração e avaliação. Se a intenção for usar o dataset inteiro,
   confirmar antes que ele cabe; uma seleção menor deve ser explícita,
   justificada e registrada, nunca consequência silenciosa de falta de espaço.
3. Guardar no Drive uma cópia persistente dos dados aprovados e um manifesto
   com versão, origem, split, rótulo, tamanho e hashes. Não versionar imagens no
   Git. Manter treino, seleção, calibração e teste em partições distintas;
   preservar o conjunto externo independente fora de treino, calibração e
   escolha de hiperparâmetros.
4. Em cada sessão do Colab, copiar automaticamente a parte necessária do Drive
   para `/content` **antes** de usá-la. Conferir espaço livre, contagem e hashes
   da cópia; falhar claramente se algo estiver ausente ou diferente. A execução
   deve ler imagens locais, enquanto checkpoints, histórico, métricas e
   resultados continuam sendo salvos no Drive para permitir retomada.
5. Após desconexão, remontar o Drive e refazer a cópia local antes de retomar.
   Não presumir que `/content` persista entre sessões. Se a cópia ou o dataset
   completo não couberem, parar e rever o escopo do novo dataset; não voltar
   automaticamente ao streaming nem reutilizar divisões de forma indevida.

## Por que mudar em relação ao OpenFake?

Na preparação da Fase 3.4, o código percorreu o OpenFake por streaming para
**selecionar e salvar** imagens no Drive. Isso evitou baixar o dataset inteiro
quando o experimento só precisava de uma amostra, mas exigiu varreduras remotas
longas e sujeitas à rede. Importante: o treino atual **não** lê exemplos do
Hugging Face por streaming; ele usa os arquivos já persistidos no Drive.

Para datasets novos, preparar primeiro uma cópia completa **do recorte
experimental definido** torna as leituras repetidas do treino e das
validações independentes da rede. Copiar esse recorte do Drive para o disco
local do Colab evita muitas operações pequenas pelo Drive montado. O custo é
espaço e tempo de preparação em cada sessão: o disco de `/content` é efêmero,
portanto o Drive continua sendo a fonte persistente. A vantagem só se aplica
quando o recorte cabe nos limites reais da conta e do runtime; não há garantia
de que qualquer dataset completo caiba.

Esta política **não autoriza** alterar a Fase 3.4, refazer a coleta OpenFake,
substituir manifestos ou iniciar treinamentos automaticamente.

Referências: [FAQ do Google Colab](https://research.google.com/colaboratory/faq.html)
(runtime temporário e limites de E/S do Drive) e
[documentação do Hugging Face Datasets](https://huggingface.co/docs/datasets/package_reference/loading_methods)
(diferença entre download local e `streaming=True`).
