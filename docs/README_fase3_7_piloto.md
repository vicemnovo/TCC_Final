# Fase 3.7 — piloto de fotografias com origem rastreável

## Por que testar esta fonte

Na seleção **interna** da Fase 3.5, todas as fotos reais do recorte Community
Forensics–Small tinham `real_source=n/a`. Também houve associação entre classe
e resolução processada. Esses fatos motivam testar uma fonte adicional de
fotografias reais com autoria declarada, antes de definir qualquer novo treino.
O resultado do Mirage-Test não seleciona imagens nem parâmetros desta etapa.

O [Open Images V7 oficial](https://storage.googleapis.com/openimages/web/download_v7.html)
publica metadados com ID, URL de origem, página da imagem, licença e perfil
do autor. Também oferece [download por ID](https://raw.githubusercontent.com/openimages/dataset/master/downloader.py)
do bucket público. O piloto usa apenas o CSV oficial do subconjunto de treino
com anotações de caixas, não o acervo completo. Filtra páginas e perfis do
Flickr com licença Creative Commons BY declarada. Isso **permite rastrear a
origem informada**, mas não comprova que cada foto foi capturada sem edição
digital ou IA. Os termos individuais ainda precisam ser respeitados; nenhuma
imagem entra no Git.

## O que o piloto faz

O script [`fase3_7_openimages_pilot.py`](../experimentos/fase3_7_openimages_pilot.py)
baixa integralmente `train-images-boxable-with-rotation.csv` para `/content`
e registra seu SHA-256. Recusa CSV acima de 1 GB e começa apenas com ao menos
2 GB livres no disco local. Conta fotos elegíveis e autores distintos,
seleciona **três IDs determinísticos para o CSV baixado** de autores distintos
com semente estável e baixa
esses três JPEGs do bucket público sem credenciais AWS. Confere leitura,
dimensões, tamanho e SHA-256. O resumo fica somente em
`/content/fase3_7_openimages_pilot/pilot_summary.json`.

O piloto **não** define treino, seleção ou calibração; não escreve no Drive,
não acessa checkpoints e não executa inferência. A seleção para eventual
treinamento será planejada e congelada em versão separada **após** conferir
a viabilidade técnica e o espaço. `/content` é temporário; repetir o piloto
após desconexão pode exigir baixar o CSV de novo. Não há streaming de imagens
durante treino ou avaliação, conforme a [política de dados](POLITICA_DADOS_FUTUROS.md).

## Execução manual no Colab

1. Abra [`piloto_fase3_7_openimages_colab.ipynb`](../Notebooks/fase_3/piloto_fase3_7_openimages_colab.ipynb)
   no VS Code e conecte um runtime Colab. **CPU é suficiente; não precisa de
   GPU.** O notebook não inicia o piloto ao abrir.
2. Execute as células **1–5** em ordem: dependências, Drive, senha de
   desbloqueio da credencial GitHub já existente, snapshot da branch e logs.
   Nenhum token ou senha é incluído no notebook ou impresso.
3. Execute a célula **6 — PILOTO MANUAL**. Observe os logs de MB baixados,
   linhas varridas, número de autorias e os três JPEGs legíveis. Se falhar,
   preserve a mensagem de erro; não selecione outro dataset automaticamente.
4. Execute a célula **7** para conferir o resumo local. Para prosseguir no
   desenvolvimento, informe o SHA do CSV, candidatas elegíveis, autores
   distintos, dimensões/tamanhos dos três JPEGs e espaço livre de `/content`
   e Drive. Não é necessário enviar as imagens.

## Gate antes do experimento completo

Com o piloto aprovado, definir previamente quantidade, teto de bytes e
distribuição das fotos reais; dividir por **autor**, sem o mesmo autor em
treino, seleção e calibração; conferir hashes contra todas as divisões
internas e benchmarks conhecidos; normalizar formatos das duas classes;
persistir manifesto e cópia no Drive; testar cópia local e retomada.
Somente então criar uma versão de ajuste fino isolada. Reservar outro teste
externo ainda não visto para a nova versão. A ausência de duplicata exata
não exclui imagens derivadas da mesma fonte.
