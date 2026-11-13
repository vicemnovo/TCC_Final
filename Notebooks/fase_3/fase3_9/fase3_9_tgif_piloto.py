"""Piloto TGIF: baixa dois arquivos completos e confere pares sem treinar.

Usa a divisao validation da fonte somente para provar acesso e estrutura.
Nao extrai arquivos, nao cria particoes do TCC e nao usa GPU ou testes externos.
"""

import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import tarfile
import time
import urllib.error
import urllib.request


BASE = 'https://cloud.ilabt.imec.be/public.php/dav/files/xEeAzrY7ES9KA8o'
SOURCE_COMMIT = 'fc5c557bd1a2d49a5dabc3676a54cfc44798f4ea'
ARCHIVES = {
    'original': ('orig/orig_validation.tar.gz', 859_947_874),
    'sd2_fr': ('sd2-fr/sd2-fr_validation.tar.gz', 895_202_795),
}
TRAIN_ARCHIVES = {
    'tgif_original': ('https://cloud.ilabt.imec.be/public.php/dav/files/xEeAzrY7ES9KA8o/'
                      'orig/orig_training.tar.gz', 5_652_563_073),
    'tgif_sd2_fr': ('https://cloud.ilabt.imec.be/public.php/dav/files/xEeAzrY7ES9KA8o/'
                   'sd2-fr/sd2-fr_training.tar.gz', 5_906_677_122),
    'tgif2_original_flux': ('https://cloud.ilabt.imec.be/public.php/dav/files/KG48tLZZzifC5WE/'
                            'orig-flux/orig-flux_training.tar.gz', 4_722_409_603),
    'tgif2_flux_schnell_fr': ('https://cloud.ilabt.imec.be/public.php/dav/files/KG48tLZZzifC5WE/'
                              'flux1schnell-fr/flux1schnell-fr_training.tar.gz',
                              15_134_859_684),
}
LOCAL = Path('/content/fase3_9_tgif_piloto')
MAX_SAMPLE_BYTES = 25_000_000


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def source_url(relative):
    if relative not in {value[0] for value in ARCHIVES.values()}:
        raise ValueError('Arquivo fora do piloto fixado')
    return BASE + '/' + relative


def remote_size(url):
    request = urllib.request.Request(url, method='HEAD',
                                     headers={'X-Requested-With': 'XMLHttpRequest'})
    with urllib.request.urlopen(request, timeout=40) as response:
        size = int(response.headers.get('Content-Length') or 0)
        if response.status != 200 or size <= 0:
            raise ValueError('Arquivo TGIF remoto indisponivel ou sem tamanho')
        return size


def download_archive(url, destination, expected_size):
    """Retoma apenas .part; nunca substitui um arquivo completo existente."""
    if destination.is_file():
        if destination.stat().st_size != expected_size:
            raise ValueError(f'Arquivo local completo com tamanho errado: {destination}')
        print(f'Arquivo completo preservado: {destination.name}', flush=True)
        return
    partial = destination.with_suffix(destination.suffix + '.part')
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > expected_size:
        raise ValueError('Download parcial maior que o arquivo remoto')
    if offset == expected_size:
        partial.rename(destination)
        return
    headers = {'User-Agent': 'TCC-Final-research/1.0'}
    if offset:
        headers['Range'] = f'bytes={offset}-'
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=90) as response:
        if offset:
            expected_range = f'bytes {offset}-{expected_size - 1}/{expected_size}'
            if response.status != 206 or response.headers.get('Content-Range') != expected_range:
                raise ValueError('Servidor nao confirmou a retomada byte a byte')
        elif response.status != 200:
            raise ValueError('Download TGIF nao retornou HTTP 200')
        last = time.monotonic()
        with partial.open('ab' if offset else 'xb') as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
                offset += len(chunk)
                if offset > expected_size:
                    raise ValueError('Arquivo remoto ultrapassou tamanho anunciado')
                if time.monotonic() - last >= 20:
                    print(f'{destination.name}: {offset / 1e6:.1f}/'
                          f'{expected_size / 1e6:.1f} MB', flush=True)
                    last = time.monotonic()
    if offset != expected_size:
        raise ValueError(f'Download incompleto: {destination.name} {offset}/{expected_size}')
    partial.rename(destination)
    print(f'{destination.name}: download completo ({expected_size / 1e6:.1f} MB)',
          flush=True)


def member_id(name, kind):
    """Identifica o COCO ID sem supor que todos os nomes sejam reconhecidos."""
    base = Path(name).name
    if kind == 'original':
        match = re.fullmatch(r'(\d+)_orig(?:_(?:512|1024))?\.png', base)
    else:
        match = re.fullmatch(r'(\d+)_mask_(?:bbox|segm)\.png_sd2-512_[012]\.png',
                             base)
    return match.group(1) if match else None


def scan_archive(path, kind, sample_limit=8):
    from PIL import Image

    ids, sizes, formats = Counter(), Counter(), Counter()
    unrecognized, examples, sampled, files = 0, [], 0, 0
    with tarfile.open(path, 'r:gz') as archive:
        for member in archive:
            if member.isdir():
                continue
            parts = Path(member.name).parts
            if not member.isfile() or member.name.startswith('/') or '..' in parts:
                raise ValueError('Membro inseguro ou nao regular no tar TGIF')
            files += 1
            image_id = member_id(member.name, kind)
            if image_id is None:
                unrecognized += 1
                if len(examples) < 8:
                    examples.append(member.name)
                continue
            ids[image_id] += 1
            if sampled < sample_limit and member.size <= MAX_SAMPLE_BYTES:
                stream = archive.extractfile(member)
                if stream is None:
                    raise ValueError('Imagem TGIF nao extraivel')
                with Image.open(io.BytesIO(stream.read())) as image:
                    image.load()
                    formats[str(image.format)] += 1
                    sizes[str(min(image.size))] += 1
                sampled += 1
            if files % 500 == 0:
                print(f'{kind}: {files} membros examinados', flush=True)
    return {'members': files, 'ids': ids, 'unrecognized': unrecognized,
            'unrecognized_examples': examples,
            'samples_decoded': sampled, 'sample_formats': dict(formats),
            'sample_short_sides': dict(sizes)}


def pilot(local):
    local.mkdir(parents=True, exist_ok=True)
    report_path = local / 'pilot_summary.json'
    if report_path.is_file():
        print('Piloto anterior preservado:', report_path, flush=True)
        print(report_path.read_text(encoding='utf-8'), flush=True)
        return
    sizes = {}
    training_available = {}
    for name, (url, expected) in TRAIN_ARCHIVES.items():
        actual = remote_size(url)
        if actual != expected:
            raise ValueError(f'Tamanho remoto de {name} mudou: {actual}/{expected}')
        training_available[name] = actual
        print(f'Fonte para preparo futuro: {name} | {actual / 1e9:.2f} GB',
              flush=True)
    for kind, (relative, expected) in ARCHIVES.items():
        size = remote_size(source_url(relative))
        if size != expected:
            raise ValueError(f'Tamanho remoto de {kind} mudou: {size}/{expected}')
        sizes[kind] = size
    missing = sum(size for kind, size in sizes.items()
                  if not (local / Path(ARCHIVES[kind][0]).name).is_file())
    if shutil.disk_usage(local).free < missing + 1_000_000_000:
        raise RuntimeError('Espaco insuficiente em /content para os dois arquivos '
                           'TGIF completos e margem de 1 GB')
    reports = {}
    for kind, (relative, expected) in ARCHIVES.items():
        destination = local / Path(relative).name
        download_archive(source_url(relative), destination, expected)
        print('Conferindo tar e imagens:', destination.name, flush=True)
        result = scan_archive(destination, kind)
        result['archive_bytes'] = expected
        result['archive_sha256'] = sha256(destination)
        reports[kind] = result
    shared = set(reports['original']['ids']) & set(reports['sd2_fr']['ids'])
    if not shared or min(item['samples_decoded'] for item in reports.values()) == 0:
        raise ValueError('Nao foi possivel comprovar pares TGIF legiveis')
    if any(item['unrecognized'] for item in reports.values()):
        print('Aviso: ha nomes nao reconhecidos; conferir exemplos no relatorio '
              'antes de preparar dados.', flush=True)
    summary = {'name': 'fase3_9_tgif_piloto_v1', 'source_repository_commit': SOURCE_COMMIT,
               'share': 'TGIF original + SD2 fully regenerated, validation somente para piloto',
               'archives': {kind: {key: value for key, value in result.items()
                                   if key != 'ids'} for kind, result in reports.items()},
               'original_unique_ids': len(reports['original']['ids']),
               'fake_unique_ids': len(reports['sd2_fr']['ids']),
               'shared_ids': len(shared),
               'naming_ready_for_preparation': not any(
                   item['unrecognized'] for item in reports.values()),
               'training_started': False,
               'partitions_created': False, 'external_used': False,
               'training_archive_bytes_verified_by_head': training_available}
    with report_path.open('x', encoding='utf-8') as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print('Piloto TGIF concluido; nenhum treino ou divisao foi criado.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--local', type=Path, default=LOCAL)
    pilot(parser.parse_args().local)


if __name__ == '__main__':
    main()
