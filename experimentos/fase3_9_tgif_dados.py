"""Prepara 768 pares TGIF originais/SD2-fr para a Fase 3.9.

Baixa os dois arquivos training completos para /content. Plano, arquivos
processados e hashes ficam no Drive; a copia local e refeita a cada sessao.
Nao usa imagens de validacao/teste da fonte nem predicoes externas.
"""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import tarfile
import urllib.request

from fase3_8_aigi_dados import normalize, prior_hashes
from fase3_8_aigi_piloto import sha256
from fase3_9_tgif_piloto import (SOURCE_COMMIT, TRAIN_ARCHIVES,
                                 download_archive, remote_size)


ROOT = Path('/content/drive/MyDrive/TCC_Final')
NAME = 'tgif_sd2fr_768_v1'
LOCAL = Path('/content') / NAME
CACHE = Path('/content/fase3_9_tgif_source')
COUNT = 768
SUBSETS = ('train', 'validation_selection', 'validation_calibration')
SEED = 'tcc-fase3-9-tgif-sd2fr-v1'
METADATA_BASE = ('https://raw.githubusercontent.com/IDLabMedia/tgif-dataset/'
                 + SOURCE_COMMIT + '/metadata/')
METADATA_FILES = ('orig_training.csv', 'orig_validation.csv', 'orig_testing.csv',
                  'sd2_training.csv')
MAX_MEMBER_BYTES = 25_000_000


def save_json(path, payload):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)


def download_metadata(name, cache):
    if name not in METADATA_FILES:
        raise ValueError('Metadata inesperado')
    target = cache / name
    if target.is_file():
        return target
    with urllib.request.urlopen(METADATA_BASE + name, timeout=45) as response:
        if response.status != 200:
            raise RuntimeError('Metadata TGIF indisponivel')
        content = response.read(2_000_001)
    if len(content) > 2_000_000:
        raise ValueError('Metadata excede 2 MB')
    with target.open('xb') as stream:
        stream.write(content)
    return target


def csv_rows(path):
    with path.open(newline='', encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def priority(value):
    return hashlib.sha256(f'{SEED}:{value}'.encode()).hexdigest()


def make_plan(metadata):
    real = csv_rows(metadata['orig_training.csv'])
    fake = csv_rows(metadata['sd2_training.csv'])
    other = {row['image_id'] for name in ('orig_validation.csv', 'orig_testing.csv')
             for row in csv_rows(metadata[name])}
    # Uma fotografia COCO pode ter mais de uma categoria. A unidade de
    # selecao e divisao e o ID, nao a linha do CSV.
    if len(real) != 2440 or len({row['image_id'] for row in real}) != 1558:
        raise ValueError('Metadata TGIF original training inesperado')
    if len(fake) != 14640:
        raise ValueError('Metadata TGIF SD2 training inesperado')
    possible = {(row['image_id'], row['mask_variation'], row['gen_variation_id'])
                for row in fake}
    categories_by_id = defaultdict(set)
    for row in real:
        image_id = row['image_id']
        if (image_id in other or not re.fullmatch(r'\d+', image_id) or
                not row['category'] or (image_id, 'bbox', '0') not in possible):
            raise ValueError('ID de treino se sobrepoe a outra divisao ou nao tem par SD2')
        categories_by_id[image_id].add(row['category'])
    groups = defaultdict(list)
    for image_id, categories in categories_by_id.items():
        category = min(categories, key=lambda value: priority(image_id + ':' + value))
        groups[category].append(image_id)
    for category in groups:
        groups[category].sort(key=priority)
    order = sorted(groups, key=priority)
    selected = []
    while len(selected) < COUNT:
        progressed = False
        for category in order:
            if groups[category] and len(selected) < COUNT:
                selected.append((groups[category].pop(), category))
                progressed = True
        if not progressed:
            raise ValueError('Menos IDs TGIF elegiveis que o tamanho fixado')
    # O mesmo COCO ID e suas duas classes ficam na mesma divisao.
    rows = []
    for index, (image_id, category) in enumerate(selected):
        position = index % 6
        subset = 'train' if position < 4 else (
            'validation_selection' if position == 4 else 'validation_calibration')
        rows.append({'image_id': image_id, 'category': category, 'subset': subset,
                     'original_member': f'{image_id}_orig_512.png',
                     'fake_member': f'{image_id}_mask_bbox.png_sd2-512_0.png'})
    if Counter(row['subset'] for row in rows) != {
            'train': 512, 'validation_selection': 128,
            'validation_calibration': 128}:
        raise ValueError('Contagens de divisao TGIF inesperadas')
    return {'name': NAME, 'source_repository_commit': SOURCE_COMMIT,
            'seed': SEED, 'pair_count': COUNT,
            'metadata_sha256': {name: sha256(path) for name, path in metadata.items()},
            'rows': rows}


def validate_plan(plan, metadata):
    expected = make_plan(metadata)
    if plan != expected:
        raise ValueError('Plano TGIF anterior difere do codigo/metadados atuais')


def selected_members(plan, kind):
    field = 'original_member' if kind == 'original' else 'fake_member'
    result = {row[field]: row for row in plan['rows']}
    if len(result) != COUNT:
        raise ValueError('Membros TGIF repetidos no plano')
    return result


def read_selected(archive_path, plan, kind):
    """Le tar local completo, sem extrair; devolve apenas 768 pares escolhidos."""
    target = selected_members(plan, kind)
    found = {}
    with tarfile.open(archive_path, 'r:gz') as archive:
        for member in archive:
            if member.isdir():
                continue
            if not member.isfile() or member.name.startswith('/') or '..' in Path(member.name).parts:
                raise ValueError('Membro inseguro no arquivo TGIF')
            name = Path(member.name).name
            if name not in target:
                continue
            if member.size > MAX_MEMBER_BYTES or member.size <= 0:
                raise ValueError(f'Membro TGIF invalido: {name}')
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f'Membro TGIF nao extraivel: {name}')
            raw = stream.read(MAX_MEMBER_BYTES + 1)
            if len(raw) != member.size:
                raise ValueError(f'Membro TGIF truncado: {name}')
            # O CSV training associa alguns COCO IDs a varias categorias.
            # Um tar pode repetir o mesmo nome; so aceitamos a repeticao se
            # os bytes tambem forem identicos. Conteudo divergente invalida
            # a correspondencia por ID e interrompe o preparo.
            if name in found:
                if found[name] != raw:
                    raise ValueError(f'Membro TGIF repetido com conteudo divergente: {name}')
                continue
            found[name] = raw
            if len(found) % 100 == 0:
                print(f'{kind}: {len(found)}/{len(target)} imagens selecionadas lidas',
                      flush=True)
    if set(found) != set(target):
        raise ValueError(f'{kind}: {len(found)}/{len(target)} membros encontrados')
    return found


def write_image(work, relative, content):
    target = work / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(content).hexdigest()
    if target.is_file():
        if sha256(target) != digest:
            raise ValueError(f'Imagem parcial mudou: {relative}')
    else:
        with target.open('xb') as stream:
            stream.write(content)
    return digest


def known_hashes(root):
    known = prior_hashes(root)  # Inclui OpenFake, CF, RAISE/MNW e Mirage.
    previous = root / 'dados_fase3_8/aigi_pares_256_v1/manifest.json'
    if not previous.is_file():
        raise FileNotFoundError(previous)
    for row in json.loads(previous.read_text(encoding='utf-8'))['rows']:
        known.update((row['original_sha256'], row['image_sha256']))
    return known


def prepare(args):
    if not args.root.is_dir():
        raise FileNotFoundError('Monte o Drive antes de preparar')
    final = args.root / 'dados_fase3_9' / NAME
    if final.exists():
        manifest = final / 'manifest.json'
        if manifest.is_file():
            print('Dados TGIF ja congelados; use stage:', manifest, flush=True)
            return
        raise FileExistsError(f'Pasta final incompleta: {final}')
    work = args.root / 'dados_fase3_9' / (NAME + '.incompleto')
    work.parent.mkdir(parents=True, exist_ok=True)
    args.cache.mkdir(parents=True, exist_ok=True)
    metadata = {name: download_metadata(name, args.cache) for name in METADATA_FILES}
    plan_path = work / 'plan.json'
    if plan_path.is_file():
        plan = json.loads(plan_path.read_text(encoding='utf-8'))
        validate_plan(plan, metadata)
        print('Retomando plano TGIF:', plan_path, flush=True)
    elif work.exists():
        raise FileExistsError(f'Pasta parcial sem plano: {work}')
    else:
        plan = make_plan(metadata)
        work.mkdir(exist_ok=False)
        save_json(plan_path, plan)
        print('Plano TGIF fixado:', plan_path, flush=True)
    archives = {}
    missing = 0
    for name in ('tgif_original', 'tgif_sd2_fr'):
        url, expected = TRAIN_ARCHIVES[name]
        if remote_size(url) != expected:
            raise ValueError(f'Tamanho remoto mudou: {name}')
        path = args.cache / (name + '.tar.gz')
        archives[name] = path
        if not path.is_file():
            missing += expected - (path.with_suffix('.gz.part').stat().st_size
                                   if path.with_suffix('.gz.part').exists() else 0)
    if shutil.disk_usage(args.cache).free < missing + 2_000_000_000:
        raise RuntimeError('Espaco /content insuficiente para dois arquivos training '
                           'completos TGIF e margem de 2 GB')
    if shutil.disk_usage(args.root).free < 500_000_000:
        raise RuntimeError('Drive com menos de 0,5 GB livres para pares TGIF')
    for name, path in archives.items():
        url, expected = TRAIN_ARCHIVES[name]
        download_archive(url, path, expected)
    raw_real = read_selected(archives['tgif_original'], plan, 'original')
    raw_fake = read_selected(archives['tgif_sd2_fr'], plan, 'fake')
    forbidden = known_hashes(args.root)
    seen = set()
    rows = []
    for index, item in enumerate(plan['rows'], 1):
        for label, kind, member, raw in (
                (0, 'original', item['original_member'], raw_real[item['original_member']]),
                (1, 'sd2-fr', item['fake_member'], raw_fake[item['fake_member']])):
            original_sha = hashlib.sha256(raw).hexdigest()
            jpeg, original, size = normalize(raw)
            processed_sha = hashlib.sha256(jpeg).hexdigest()
            if (original['format'] != 'PNG' or min(original['size']) < 256 or
                    original_sha in forbidden or processed_sha in forbidden or
                    original_sha in seen or processed_sha in seen):
                raise ValueError(f'Duplicata, formato ou tamanho invalido: {item["image_id"]}')
            seen.update((original_sha, processed_sha))
            relative = f'images/{label}/{item["image_id"]}.jpg'
            if write_image(work, relative, jpeg) != processed_sha:
                raise ValueError('Falha ao gravar imagem normalizada')
            rows.append({'image_id': item['image_id'], 'label': label,
                         'subset': item['subset'], 'category': item['category'],
                         'group': 'coco' if label == 0 else 'sd2-fr',
                         'kind': kind, 'source_member': member, 'image': relative,
                         'image_sha256': processed_sha, 'original_sha256': original_sha,
                         'original_format': original['format'],
                         'original_size': original['size'], 'processed_size': size})
        if index % 100 == 0 or index == len(plan['rows']):
            print(f'Pares TGIF: {index}/{len(plan["rows"])} persistidos', flush=True)
    counts = Counter((row['subset'], row['label']) for row in rows)
    if any(counts[subset, label] != size for subset, size in
           (('train', 512), ('validation_selection', 128),
            ('validation_calibration', 128)) for label in (0, 1)):
        raise ValueError('Contagens de pares TGIF incorretas')
    manifest = {'name': NAME, 'source': 'IDLabMedia/tgif-dataset',
                'source_repository_commit': SOURCE_COMMIT,
                'source_archives': {key: {'bytes': TRAIN_ARCHIVES[key][1],
                                         'sha256': sha256(path)}
                                    for key, path in archives.items()},
                'plan_sha256': sha256(plan_path),
                'preprocessing': 'RGB; lado curto 256 LANCZOS; JPEG qualidade 90',
                'rows': rows}
    manifest_path = work / 'manifest.json'
    if manifest_path.is_file():
        if json.loads(manifest_path.read_text(encoding='utf-8')) != manifest:
            raise ValueError('Manifesto TGIF parcial difere da verificacao atual')
    else:
        save_json(manifest_path, manifest)
    work.rename(final)
    print('Dados TGIF congelados:', final, '| contagens:', dict(counts), flush=True)


def stage(args):
    source = args.root / 'dados_fase3_9' / NAME
    manifest_path = source / 'manifest.json'
    if not manifest_path.is_file():
        raise FileNotFoundError(f'Execute prepare primeiro: {manifest_path}')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('name') != NAME or len(manifest['rows']) != COUNT * 2:
        raise ValueError('Manifesto TGIF incompleto')
    args.local.mkdir(parents=True, exist_ok=True)
    local_manifest = args.local / 'manifest.json'
    if local_manifest.exists() and sha256(local_manifest) != sha256(manifest_path):
        raise ValueError('Manifesto local difere do Drive')
    for index, row in enumerate(manifest['rows'], 1):
        relative = Path(row['image'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Caminho inseguro no manifesto')
        original, target = source / relative, args.local / relative
        if not original.is_file() or sha256(original) != row['image_sha256']:
            raise ValueError(f'Imagem TGIF no Drive mudou: {relative}')
        if target.exists():
            if sha256(target) != row['image_sha256']:
                raise ValueError(f'Imagem TGIF local mudou: {relative}')
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, target)
            if sha256(target) != row['image_sha256']:
                raise ValueError('Falha na copia local TGIF')
        if index % 100 == 0 or index == len(manifest['rows']):
            print(f'Copia TGIF: {index}/{len(manifest["rows"])}', flush=True)
    if not local_manifest.is_file():
        shutil.copy2(manifest_path, local_manifest)
    print('Dados TGIF locais prontos:', args.local, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'stage'))
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--local', type=Path, default=LOCAL)
    parser.add_argument('--cache', type=Path, default=CACHE)
    args = parser.parse_args()
    (prepare if args.mode == 'prepare' else stage)(args)


if __name__ == '__main__':
    main()
