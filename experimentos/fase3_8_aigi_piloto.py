"""Piloto AIGI: verifica pares e imagens reais obtíveis, sem criar treino.

Baixa o CSV e um shard Parquet completos para /content antes de examiná-los.
Baixa cada fotografia de prova por inteiro antes de abri-la. Não usa streaming
de exemplos, datasets externos de avaliação, checkpoints ou Google Drive.
"""

import argparse
from collections import Counter
import csv
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import shutil
import socket
import tempfile
import urllib.error
import urllib.parse
import urllib.request


REPO = 'Coxy7/AIGI-Detection-Quality-Paradox'
SHARD = 'data/fake-00000-of-00015.parquet'
REAL_CSV = 'real_images_metadata.csv'
LOCAL = Path('/content/fase3_8_aigi_piloto')
SEED = 'tcc-fase3-8-aigi-piloto-v1'
REAL_SOURCES = {'Laion-2B-en-aesthetic', 'CC3M'}
MAX_IMAGE_BYTES = 15_000_000


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def priority(uid):
    return hashlib.sha256(f'{SEED}:{uid}'.encode('utf-8')).hexdigest()


def pinned_files(local):
    from huggingface_hub import HfApi, hf_hub_download

    info = HfApi(token=os.environ.get('HF_TOKEN') or False).dataset_info(REPO)
    if not info.sha or len(info.sha) != 40:
        raise RuntimeError('Commit do AIGI indisponível')
    files = {}
    for name in (REAL_CSV, SHARD):
        print(f'Baixando arquivo completo: {name} | commit={info.sha}', flush=True)
        files[name] = Path(hf_hub_download(
            repo_id=REPO, repo_type='dataset', filename=name,
            revision=info.sha, local_dir=local / 'source',
            token=os.environ.get('HF_TOKEN') or False))
        print(f'Arquivo pronto: {name} | {files[name].stat().st_size / 1e6:.1f} MB', flush=True)
    return info.sha, files


def real_rows(path):
    result, sources = {}, Counter()
    with path.open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        required = {'uid', 'source_dataset', 'image_path'}
        if not required <= set(reader.fieldnames or ()):
            raise ValueError(f'Colunas reais ausentes: {required - set(reader.fieldnames or ())}')
        for row in reader:
            uid = (row['uid'] or '').strip()
            if not uid or uid in result:
                raise ValueError(f'UID real vazio ou repetido: {uid!r}')
            result[uid] = row
            sources[row['source_dataset']] += 1
    if len(result) != 4000:
        raise ValueError(f'Esperados 4.000 UIDs reais distintos; encontrados {len(result)}')
    return result, sources


def fake_metadata(path):
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    fields = {'uid', 'generator', 'format', 'height', 'width', 'image'}
    if not fields <= set(parquet.schema_arrow.names):
        raise ValueError(f'Colunas sintéticas ausentes: {fields - set(parquet.schema_arrow.names)}')
    rows, generators, formats, sides = [], Counter(), Counter(), []
    for batch in parquet.iter_batches(
            columns=['uid', 'generator', 'format', 'height', 'width'], batch_size=512):
        for row in batch.to_pylist():
            uid = str(row['uid'] or '').strip()
            generator = str(row['generator'] or '').strip()
            height, width = int(row['height']), int(row['width'])
            if not uid or not generator or min(height, width) < 224:
                raise ValueError('Metadados sintéticos inválidos')
            rows.append({'uid': uid, 'generator': generator})
            generators[generator] += 1
            formats[str(row['format'])] += 1
            sides.append(min(height, width))
    if not rows:
        raise ValueError('Shard sintético vazio')
    sides.sort()
    quantiles = [sides[int((len(sides)-1)*p)] for p in (0.1, 0.5, 0.9)]
    return rows, generators, formats, quantiles


def first_fake_image(path):
    """Confere que os bytes de pelo menos uma imagem do shard são legíveis."""
    import io
    from PIL import Image
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    batch = next(parquet.iter_batches(columns=['image'], batch_size=1))
    value = batch.to_pylist()[0]['image']
    raw = value.get('bytes') if isinstance(value, dict) else value
    if not isinstance(raw, bytes):
        raise ValueError('Imagem sintética não contém bytes no shard')
    with Image.open(io.BytesIO(raw)) as image:
        image.load()
        return {'format': image.format, 'size': list(image.size),
                'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def public_https_url(value):
    parsed = urllib.parse.urlparse(value)
    if (parsed.scheme != 'https' or not parsed.hostname or '.' not in parsed.hostname
            or parsed.username or parsed.password or parsed.port not in (None, 443)):
        raise ValueError('URL pública HTTPS necessária')
    for answer in socket.getaddrinfo(parsed.hostname, 443, type=socket.SOCK_STREAM):
        address = ipaddress.ip_address(answer[4][0])
        if not address.is_global:
            raise ValueError('URL não resolve para endereço público')
    return value


class PublicRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        public_https_url(newurl)
        return super().redirect_request(request, fp, code, msg, headers, newurl)


def download_probe(row, local):
    """Tenta uma foto da fonte; erros são resumidos sem expor a URL."""
    from PIL import Image, ImageOps

    url = public_https_url(row['image_path'])
    target = local / 'real_probe' / (row['uid'] + '.img')
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.is_file():
        opener = urllib.request.build_opener(PublicRedirects())
        request = urllib.request.Request(url, headers={'User-Agent': 'TCC-Final-research/1.0'})
        with opener.open(request, timeout=20) as response:
            length = int(response.headers.get('Content-Length') or 0)
            if response.status != 200 or length > MAX_IMAGE_BYTES:
                raise ValueError('HTTP ou tamanho da imagem inválido')
            with tempfile.NamedTemporaryFile(dir=target.parent, suffix='.part', delete=False) as output:
                temporary = Path(output.name)
                size = 0
                try:
                    while block := response.read(1024 * 1024):
                        size += len(block)
                        if size > MAX_IMAGE_BYTES:
                            raise ValueError('Imagem excedeu o teto de 15 MB')
                        output.write(block)
                except BaseException:
                    output.close()
                    temporary.unlink(missing_ok=True)
                    raise
            temporary.replace(target)
    with Image.open(target) as opened:
        image = ImageOps.exif_transpose(opened)
        image.load()
        short_side = min(image.size)
        result = {'uid': row['uid'], 'source': row['source_dataset'],
                  'format': opened.format, 'size': list(image.size),
                  'bytes': target.stat().st_size, 'sha256': sha256(target),
                  'short_side_at_least_512': short_side >= 512}
    return result


def pilot(args):
    if shutil.disk_usage(args.local.parent).free < 2_000_000_000:
        raise RuntimeError('/content com menos de 2 GB livres para o piloto AIGI')
    args.local.mkdir(parents=True, exist_ok=True)
    revision, files = pinned_files(args.local)
    real, sources = real_rows(files[REAL_CSV])
    fake, generators, formats, sides = fake_metadata(files[SHARD])
    fake_uids = {row['uid'] for row in fake}
    if len(fake_uids) != len(fake):
        print('Aviso: há UIDs repetidos entre geradores; dividir por UID.', flush=True)
    eligible = [real[uid] for uid in fake_uids & real.keys()
                if real[uid]['source_dataset'] in REAL_SOURCES
                and urllib.parse.urlparse(real[uid]['image_path']).scheme == 'https']
    eligible.sort(key=lambda row: priority(row['uid']))
    print('Metadados reais:', dict(sources), flush=True)
    print('Shard sintético:', len(fake), '| geradores:', dict(generators),
          '| formatos:', dict(formats), '| lado curto p10/med/p90:', sides, flush=True)
    print('UIDs pareáveis com URL HTTPS neste shard:', len(eligible), flush=True)
    if not eligible:
        raise ValueError('Nenhum par com fotografia real acessível por URL HTTPS')
    outcomes, failures = [], Counter()
    for index, row in enumerate(eligible[:args.samples], 1):
        try:
            result = download_probe(row, args.local)
            outcomes.append(result)
            print(f'Foto real piloto: {index}/{min(args.samples,len(eligible))} '
                  f'| legível | lado curto={min(result["size"])}', flush=True)
        except (OSError, ValueError, urllib.error.URLError) as error:
            failures[type(error).__name__] += 1
            print(f'Foto real piloto: {index}/{min(args.samples,len(eligible))} '
                  f'| indisponível ({type(error).__name__})', flush=True)
    report = {'source': REPO, 'revision': revision,
              'csv_sha256': sha256(files[REAL_CSV]),
              'shard': SHARD, 'shard_sha256': sha256(files[SHARD]),
              'real_metadata_count': len(real), 'real_sources': dict(sources),
              'fake_in_shard': len(fake), 'fake_generators': dict(generators),
              'fake_formats': dict(formats), 'fake_short_side_p10_med_p90': sides,
              'fake_first_image': first_fake_image(files[SHARD]),
              'paired_https_candidates_in_shard': len(eligible),
              'real_probe_attempted': min(args.samples, len(eligible)),
              'real_probe_success': len(outcomes), 'real_probe_failures': dict(failures),
              'real_probe_short_side_512_plus': sum(
                  row['short_side_at_least_512'] for row in outcomes),
              'real_probe': outcomes,
              'train_or_evaluation_started': False}
    path = args.local / 'pilot_summary.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8')) != report:
        raise ValueError('Resumo do piloto anterior diverge; não sobrescrever')
    if not path.exists():
        with path.open('x', encoding='utf-8') as stream:
            json.dump(report, stream, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    print('Piloto concluído: nenhuma divisão ou treino criado.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('pilot',))
    parser.add_argument('--local', type=Path, default=LOCAL)
    parser.add_argument('--samples', type=int, default=16)
    args = parser.parse_args()
    if not 1 <= args.samples <= 40:
        raise ValueError('Escolha de 1 a 40 fotos reais para o piloto')
    pilot(args)


if __name__ == '__main__':
    main()
