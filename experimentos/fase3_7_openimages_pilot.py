"""Piloto somente diagnóstico de fotos com autor rastreável no Open Images.

Baixa o CSV oficial completo e três JPEGs para /content. Não cria divisões,
não escreve no Drive e não lê checkpoints ou resultados de avaliação.
"""

import argparse
import csv
import hashlib
import heapq
import json
from pathlib import Path
import re
import shutil
import time
import urllib.parse
import urllib.request


METADATA_URL = (
    'https://storage.googleapis.com/openimages/2018_04/train/'
    'train-images-boxable-with-rotation.csv'
)
BUCKET = 'open-images-dataset'
LOCAL = Path('/content/fase3_7_openimages_pilot')
SEED = 'tcc-fase3-7-openimages-pilot-v1'
MAX_CSV_BYTES = 1_000_000_000
REQUIRED_COLUMNS = {
    'ImageID', 'Subset', 'OriginalURL', 'OriginalLandingURL', 'License',
    'AuthorProfileURL', 'Author', 'OriginalSize', 'OriginalMD5',
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def download_metadata(target):
    """Guarda o CSV inteiro antes de ler linhas; nunca usa exemplos por stream."""
    if target.is_file():
        print(f'CSV local existente: {target} ({target.stat().st_size:,} bytes)', flush=True)
        return target
    if shutil.disk_usage(target.parent).free < 2_000_000_000:
        raise RuntimeError('/content com menos de 2 GB livres para o piloto')
    request = urllib.request.Request(METADATA_URL, headers={'User-Agent': 'TCC-Final-research/1.0'})
    with urllib.request.urlopen(request, timeout=90) as response:
        length = int(response.headers.get('Content-Length') or 0)
        if response.status != 200 or length > MAX_CSV_BYTES:
            raise RuntimeError(f'CSV inesperado: HTTP {response.status}, tamanho {length}')
        temporary = target.with_suffix('.csv.part')
        if temporary.exists():
            # Só este arquivo temporário incompleto é substituído. O CSV já
            # concluído acima nunca é rebaixado ou sobrescrito.
            temporary.unlink()
            print('Download parcial anterior descartado; reiniciando CSV.', flush=True)
        count, last = 0, time.monotonic()
        with temporary.open('xb') as output:
            while block := response.read(1024 * 1024):
                count += len(block)
                if count > MAX_CSV_BYTES:
                    raise RuntimeError('CSV excedeu o limite de 1 GB; interrompa o piloto')
                output.write(block)
                if time.monotonic() - last >= 20:
                    print(f'CSV: {count / 1e6:.1f} MB baixados', flush=True)
                    last = time.monotonic()
        if length and count != length:
            raise RuntimeError(f'CSV incompleto: {count}/{length} bytes')
        temporary.replace(target)
    print(f'CSV completo: {count / 1e6:.1f} MB | SHA-256={sha256(target)}', flush=True)
    return target


def normalized_author(value):
    parsed = urllib.parse.urlparse((value or '').strip())
    if parsed.scheme not in ('http', 'https') or parsed.hostname not in (
            'flickr.com', 'www.flickr.com'):
        return None
    parts = [part for part in parsed.path.lower().split('/') if part]
    if len(parts) < 2 or parts[0] != 'people':
        return None
    return f'flickr:{parts[1]}'


def eligible(row):
    """Aceita apenas fotos do treino com perfil Flickr e licença CC BY.

    Esses metadados rastreiam a fonte declarada, mas não certificam a captura
    nem excluem montagem ou edição digital. Não inferir autenticidade daqui.
    """
    image_id = (row.get('ImageID') or '').lower()
    author = normalized_author(row.get('AuthorProfileURL'))
    landing = urllib.parse.urlparse((row.get('OriginalLandingURL') or '').strip())
    license_url = urllib.parse.urlparse((row.get('License') or '').strip())
    license_parts = [part for part in license_url.path.lower().split('/') if part]
    if (row.get('Subset') != 'train' or not re.fullmatch(r'[0-9a-f]{16}', image_id)
            or author is None or landing.hostname not in ('flickr.com', 'www.flickr.com')
            or not landing.path.startswith('/photos/')
            or license_url.hostname != 'creativecommons.org'
            or license_parts[:2] != ['licenses', 'by']):
        return None
    return {'image_id': image_id, 'author_group': author,
            'license': row['License'], 'landing_url': row['OriginalLandingURL']}


def scan_metadata(path, sample_count=3):
    """Escolhe amostras determinísticas de autores distintos, sem predições."""
    if sample_count < 1 or sample_count > 10:
        raise ValueError('Piloto limitado a 1–10 imagens')
    count, authors, heap = 0, set(), []
    number = 0
    with Path(path).open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        if not REQUIRED_COLUMNS <= set(reader.fieldnames or ()):
            raise ValueError('Colunas oficiais esperadas ausentes no CSV')
        for number, row in enumerate(reader, 1):
            candidate = eligible(row)
            if candidate:
                count += 1
                authors.add(candidate['author_group'])
                priority = int(hashlib.sha256(
                    f"{SEED}:{candidate['image_id']}".encode()).hexdigest(), 16)
                # A lista curta reduz memória; o conjunto de autores mede
                # quantos grupos separados podem existir antes de selecionar.
                item = (-priority, candidate['image_id'], number, candidate)
                if len(heap) < 100:
                    heapq.heappush(heap, item)
                elif item > heap[0]:
                    heapq.heapreplace(heap, item)
            if number % 250_000 == 0:
                print(f'Metadados: {number:,} linhas | elegíveis={count:,} '
                      f'| autores={len(authors):,}', flush=True)
    selected, seen = [], set()
    for _, _, _, candidate in sorted(heap, reverse=True):
        if candidate['author_group'] in seen:
            continue
        selected.append(candidate)
        seen.add(candidate['author_group'])
        if len(selected) == sample_count:
            break
    if len(selected) != sample_count:
        raise ValueError('Autores distintos insuficientes para o piloto')
    return {'metadata_rows': number, 'eligible_images': count,
            'distinct_authors': len(authors), 'pilot_rows': selected}


def sample_images(rows, local_dir):
    """Baixa somente os IDs do piloto do bucket público sem credenciais AWS."""
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config
    from PIL import Image, ImageOps

    bucket = boto3.resource('s3', config=Config(signature_version=UNSIGNED)).Bucket(BUCKET)
    output = []
    for row in rows:
        image_id = row['image_id']
        target = local_dir / f'{image_id}.jpg'
        if not target.exists():
            partial = target.with_suffix('.jpg.part')
            if partial.exists():
                partial.unlink()  # Somente o download parcial do piloto.
            bucket.download_file(f'train/{image_id}.jpg', str(partial))
            partial.replace(target)
        with Image.open(target) as opened:
            image = ImageOps.exif_transpose(opened)
            image.load()
            dimensions = list(image.size)
            image_format = opened.format
        if image_format != 'JPEG':
            raise ValueError(f'Formato inesperado para {image_id}: {image_format}')
        output.append({'image_id': image_id, 'author_group_sha256': hashlib.sha256(
            row['author_group'].encode()).hexdigest(), 'bytes': target.stat().st_size,
            'width': dimensions[0], 'height': dimensions[1],
            'image_sha256': sha256(target)})
        print(f'Piloto: {len(output)}/{len(rows)} JPEGs legíveis', flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--local-dir', type=Path, default=LOCAL)
    parser.add_argument('--samples', type=int, default=3)
    args = parser.parse_args()
    args.local_dir.mkdir(parents=True, exist_ok=True)
    csv_path = download_metadata(args.local_dir / 'train-images-boxable-with-rotation.csv')
    summary = scan_metadata(csv_path, args.samples)
    summary['metadata_url'] = METADATA_URL
    summary['metadata_sha256'] = sha256(csv_path)
    summary['sample_images'] = sample_images(summary['pilot_rows'], args.local_dir)
    report = args.local_dir / 'pilot_summary.json'
    if report.exists():
        if json.loads(report.read_text(encoding='utf-8')) != summary:
            raise ValueError('Relatório do piloto anterior difere; preserve os dois antes de prosseguir')
    else:
        with report.open('x', encoding='utf-8') as stream:
            json.dump(summary, stream, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print('Piloto concluído: nenhuma divisão, seleção de treino ou checkpoint foi criado.',
          flush=True)


if __name__ == '__main__':
    main()
