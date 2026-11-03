"""Prepara pares AIGI reais/sintéticos para a Fase 3.8, sem treinar.

Baixa seis shards Parquet e o CSV inteiros em /content. A seleção, falhas de
download, hashes e divisões são persistidos no Drive para retomada. Imagens
externas só entram na conferência de hashes; predições externas nunca são lidas.
"""

import argparse
from collections import Counter
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil

from fase3_8_aigi_piloto import (REPO, REAL_CSV, download_probe,
                                  priority, real_rows, sha256)


ROOT = Path('/content/drive/MyDrive/TCC_Final')
NAME = 'aigi_pares_256_v1'
LOCAL = Path('/content') / NAME
CACHE = Path('/content/fase3_8_aigi_source')
REAL_CACHE = Path('/content/fase3_8_aigi_real_cache')
PER_GENERATOR = 80
SIDE = 256
REVISION = '9244882a51dbe33e658fe514488692155d20e5dd'
SHARDS = {
    'FLUX.1_dev': 'data/fake-00000-of-00015.parquet',
    'Infinity': 'data/fake-00003-of-00015.parquet',
    'PA_alpha': 'data/fake-00005-of-00015.parquet',
    'SDXL_1.0': 'data/fake-00008-of-00015.parquet',
    'SD_2.1': 'data/fake-00010-of-00015.parquet',
    'SD_3': 'data/fake-00013-of-00015.parquet',
}
SUBSETS = {'PA_alpha': 'validation_selection',
           'SD_2.1': 'validation_calibration'}


def atomic_json(path, payload):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temp, path)


def real_url(row):
    source = row['source_dataset']
    if source == 'MSCOCO':
        name = row['image_path']
        if not name.startswith('COCO_val2014_') or not name.endswith('.jpg') or '/' in name:
            return None
        return 'https://s3.amazonaws.com/images.cocodataset.org/val2014/' + name
    if source in ('Laion-2B-en-aesthetic', 'CC3M') and row['image_path'].startswith('https://'):
        return row['image_path']
    return None


def candidate_rows(shards, real):
    import pyarrow.parquet as pq

    candidates = {}
    for generator, path in shards.items():
        parquet = pq.ParquetFile(path)
        required = {'uid', 'generator', 'height', 'width', 'image'}
        if not required <= set(parquet.schema_arrow.names):
            raise ValueError(f'Esquema AIGI incompleto em {path.name}')
        found, count = [], 0
        for batch in parquet.iter_batches(
                columns=['uid', 'generator', 'height', 'width'], batch_size=512):
            for row in batch.to_pylist():
                if row['generator'] != generator:
                    raise ValueError(f'{path.name} contém outro gerador; plano de shards inválido')
                uid = str(row['uid'])
                if not re.fullmatch(r'[A-Za-z0-9_-]{8}', uid):
                    raise ValueError('UID AIGI inesperado')
                source = real.get(uid)
                if source and real_url(source) and min(int(row['height']), int(row['width'])) >= SIDE:
                    found.append({'uid': uid, 'generator': generator,
                                  'shard': path.name, 'index': count,
                                  'real_source': source['source_dataset'],
                                  'real_url': real_url(source)})
                count += 1
        found.sort(key=lambda row: priority(f"{row['generator']}:{row['uid']}"))
        candidates[generator] = found
        print(f'{generator}: {count} sintéticas no shard, '
              f'{len(found)} UIDs pareáveis com URL real', flush=True)
        if len(found) < PER_GENERATOR:
            raise ValueError(f'{generator}: menos de {PER_GENERATOR} candidatos antes do download real')
    return candidates


def normalize(raw):
    from PIL import Image, ImageOps

    with Image.open(io.BytesIO(raw)) as opened:
        original = {'format': opened.format, 'size': list(opened.size)}
        image = ImageOps.exif_transpose(opened).convert('RGB')
    if min(image.size) < SIDE:
        raise ValueError(f'Imagem original menor que {SIDE} px; não ampliar')
    scale = SIDE / min(image.size)
    image = image.resize((round(image.width * scale), round(image.height * scale)),
                         Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', quality=90, subsampling=0)
    return buffer.getvalue(), original, list(image.size)


def prior_hashes(root):
    from fase3_7_dados import known_hashes

    known = known_hashes(root)
    previous = root / 'dados_fase3_7/fase3_7_cfsmall_expansao_v1/manifest.json'
    if not previous.is_file():
        raise FileNotFoundError(previous)
    for row in json.loads(previous.read_text(encoding='utf-8'))['rows']:
        known.add(row['original_sha256'])
        known.add(row['image_sha256'])
    return known


def safe_write(work, rel, content):
    path = work / rel
    value = hashlib.sha256(content).hexdigest()
    if path.exists():
        if sha256(path) != value:
            raise ValueError(f'Arquivo alterado no Drive: {rel}')
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(content)
    return value


def load_plan(args):
    from huggingface_hub import hf_hub_download

    final = args.root / 'dados_fase3_8' / NAME
    work = args.root / 'dados_fase3_8' / (NAME + '.incompleto')
    if final.exists():
        raise FileExistsError(f'Dados 3.8 já congelados: {final}')
    work.parent.mkdir(parents=True, exist_ok=True)
    plan_path = work / 'plan.json'
    if plan_path.is_file():
        plan = json.loads(plan_path.read_text(encoding='utf-8'))
        if (plan.get('name') != NAME or plan.get('shards') != SHARDS
                or plan.get('per_generator') != PER_GENERATOR or plan.get('side') != SIDE
                or plan.get('revision') != REVISION):
            raise ValueError('Plano 3.8 anterior não corresponde ao código atual')
        revision = plan['revision']
        print('Retomando plano:', plan_path, flush=True)
    elif work.exists():
        raise FileExistsError(f'Pasta incompleta sem plano: {work}')
    else:
        revision = REVISION
        plan = None
    args.cache.mkdir(parents=True, exist_ok=True)
    files = {}
    for name in (REAL_CSV, *SHARDS.values()):
        print(f'Baixando/verificando arquivo completo: {name}', flush=True)
        files[name] = Path(hf_hub_download(
            repo_id=REPO, repo_type='dataset', filename=name,
            revision=revision, local_dir=args.cache,
            token=os.environ.get('HF_TOKEN') or False))
    if plan is None:
        real, _ = real_rows(files[REAL_CSV])
        candidates = candidate_rows({group: files[shard] for group, shard in SHARDS.items()}, real)
        plan = {'name': NAME, 'source': REPO, 'revision': revision,
                'side': SIDE, 'per_generator': PER_GENERATOR,
                'shards': SHARDS, 'source_sha256': {name: sha256(path) for name, path in files.items()},
                'candidates': candidates,
                'selection_generator': 'PA_alpha',
                'calibration_generator': 'SD_2.1'}
        work.mkdir(exist_ok=False)
        with plan_path.open('x', encoding='utf-8') as stream:
            json.dump(plan, stream, ensure_ascii=False, indent=2)
        print('Plano de candidatos fixado:', plan_path, flush=True)
    for name, path in files.items():
        if sha256(path) != plan['source_sha256'][name]:
            raise ValueError(f'Arquivo da fonte mudou: {name}')
    return work, plan, files


def prepare(args):
    if not args.root.is_dir():
        raise FileNotFoundError('Monte o Google Drive antes do preparo')
    if shutil.disk_usage(args.cache.parent).free < 5_000_000_000:
        raise RuntimeError('/content com menos de 5 GB livres para os seis shards')
    if shutil.disk_usage(args.root).free < 1_000_000_000:
        raise RuntimeError('Drive com menos de 1 GB livre; não preparar')
    work, plan, files = load_plan(args)
    forbidden = prior_hashes(args.root)
    state_path = work / 'state.json'
    state = json.loads(state_path.read_text(encoding='utf-8')) if state_path.is_file() else {
        'accepted': {}, 'failed': {}}
    accepted, failed = state['accepted'], state['failed']
    used_uids = {item['uid'] for rows in accepted.values() for item in rows}
    if len(used_uids) != sum(len(rows) for rows in accepted.values()):
        raise ValueError('UID real reutilizado entre geradores')
    accepted_hashes = set()
    for rows in accepted.values():
        for item in rows:
            path = work / item['real_image']
            if not path.is_file() or sha256(path) != item['real_image_sha256']:
                raise ValueError(f"Foto real do plano ausente ou alterada: {item['uid']}")
            accepted_hashes.update((item['real_original_sha256'], item['real_image_sha256']))
    real_cache = args.real_cache
    real_cache.mkdir(parents=True, exist_ok=True)
    for generator, candidates in plan['candidates'].items():
        chosen = accepted.setdefault(generator, [])
        for item in candidates:
            if len(chosen) >= PER_GENERATOR:
                break
            uid = item['uid']
            key = f'{generator}:{uid}'
            if uid in used_uids or key in failed:
                continue
            row = {'uid': uid, 'source_dataset': item['real_source'],
                   'image_path': item['real_url']}
            try:
                probe = download_probe(row, real_cache)
                if min(probe['size']) < SIDE:
                    raise ValueError('Foto real abaixo do lado mínimo')
                original_path = real_cache / 'real_probe' / (uid + '.img')
                raw = original_path.read_bytes()
                if probe['sha256'] in forbidden or probe['sha256'] in accepted_hashes:
                    raise ValueError('Sobreposição exata com dados anteriores/reservados')
                jpeg, original, processed = normalize(raw)
                rel = f'images/0/{uid}.jpg'
                value = safe_write(work, rel, jpeg)
                if value in forbidden or value in accepted_hashes:
                    raise ValueError('Sobreposição processada com dados anteriores/reservados')
                chosen.append({**item, 'subset': SUBSETS.get(generator, 'train'),
                               'real_image': rel, 'real_original_sha256': probe['sha256'],
                               'real_image_sha256': value,
                               'real_original_format': original['format'],
                               'real_original_size': original['size'],
                               'processed_size': processed})
                used_uids.add(uid)
                accepted_hashes.update((probe['sha256'], value))
            except (OSError, ValueError) as error:
                failed[key] = type(error).__name__
            atomic_json(state_path, state)
            attempts = sum(len(rows) for rows in accepted.values()) + len(failed)
            if attempts % 25 == 0:
                print(f'Pares reais aceitos={sum(len(rows) for rows in accepted.values())}/'
                      f'{len(SHARDS)*PER_GENERATOR} | tentativas={attempts}', flush=True)
        if len(chosen) < PER_GENERATOR:
            raise RuntimeError(f'{generator}: apenas {len(chosen)}/{PER_GENERATOR} reais '
                               'obtidas; plano preservado para diagnóstico/retomada')
        print(f'{generator}: {len(chosen)} fotografias reais verificadas', flush=True)
    selected = {(item['shard'], item['index']): item
                for rows in accepted.values() for item in rows}
    if len(selected) != len(SHARDS)*PER_GENERATOR:
        raise ValueError('Seleção sintética incompleta')
    rows = []
    for generator, shard in SHARDS.items():
        import pyarrow.parquet as pq

        parquet = pq.ParquetFile(files[shard])
        index = 0
        for batch in parquet.iter_batches(columns=['image'], batch_size=16):
            for source in batch.to_pylist():
                item = selected.get((Path(shard).name, index))
                index += 1
                if item is None:
                    continue
                value = source['image']
                raw = value.get('bytes') if isinstance(value, dict) else value
                if not isinstance(raw, bytes):
                    raise ValueError('Imagem sintética sem bytes')
                original_sha = hashlib.sha256(raw).hexdigest()
                if original_sha in forbidden:
                    raise ValueError('Sintética sobrepõe dados anteriores/reservados')
                jpeg, original, processed = normalize(raw)
                fake_rel = f"images/1/{item['uid']}.jpg"
                fake_sha = safe_write(work, fake_rel, jpeg)
                if fake_sha in forbidden or fake_sha == item['real_image_sha256']:
                    raise ValueError('Sintética processada duplicada')
                for label, rel, image_sha, original_hash, image_format, original_size, size in (
                        (0, item['real_image'], item['real_image_sha256'],
                         item['real_original_sha256'], item['real_original_format'],
                         item['real_original_size'], item['processed_size']),
                        (1, fake_rel, fake_sha, original_sha,
                         original['format'], original['size'], processed)):
                    rows.append({'uid': item['uid'], 'label': label,
                                 'subset': item['subset'], 'group': generator,
                                 'real_source': item['real_source'], 'image': rel,
                                 'image_sha256': image_sha,
                                 'original_sha256': original_hash,
                                 'original_format': image_format,
                                 'original_size': original_size,
                                 'processed_size': size})
                if len(rows) % 200 == 0:
                    print(f'Pares processados: {len(rows)//2}/{len(SHARDS)*PER_GENERATOR}',
                          flush=True)
    if len(rows) != 2*len(SHARDS)*PER_GENERATOR:
        raise ValueError('Quantidade final de pares incompleta')
    seen = set()
    for row in rows:
        value = row['image_sha256']
        if value in seen:
            raise ValueError('Duplicata exata no AIGI selecionado')
        seen.add(value)
    counts = Counter((row['subset'], row['label']) for row in rows)
    print('Divisões AIGI:', dict(counts), flush=True)
    print('Formatos originais AIGI por classe:',
          {label: dict(Counter(row['original_format'] for row in rows
                               if row['label'] == label)) for label in (0, 1)},
          flush=True)
    print('Fontes reais AIGI:', dict(Counter(row['real_source'] for row in rows
                                              if row['label'] == 0)), flush=True)
    manifest = {'name': NAME, 'source': REPO, 'revision': plan['revision'],
                'plan_sha256': sha256(work / 'plan.json'),
                'preprocessing': 'RGB; lado curto 256 LANCZOS sem ampliar; JPEG qualidade 90',
                'rows': sorted(rows, key=lambda row: (row['subset'], row['uid'], row['label']))}
    with (work / 'manifest.json').open('x', encoding='utf-8') as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
    work.rename(args.root / 'dados_fase3_8' / NAME)
    print('Dados 3.8 congelados:', args.root / 'dados_fase3_8' / NAME, flush=True)


def stage(args):
    source = args.root / 'dados_fase3_8' / NAME
    manifest = source / 'manifest.json'
    if not manifest.is_file():
        raise FileNotFoundError(f'Execute prepare primeiro: {manifest}')
    rows = json.loads(manifest.read_text(encoding='utf-8'))['rows']
    if len(rows) != 2*len(SHARDS)*PER_GENERATOR:
        raise ValueError('Manifesto 3.8 incompleto')
    args.local.mkdir(parents=True, exist_ok=True)
    local_manifest = args.local / 'manifest.json'
    if local_manifest.exists() and sha256(local_manifest) != sha256(manifest):
        raise ValueError('Manifesto local difere do Drive')
    for index, row in enumerate(rows, 1):
        rel = Path(row['image'])
        if rel.is_absolute() or '..' in rel.parts:
            raise ValueError('Caminho inseguro no manifesto')
        src, dst = source / rel, args.local / rel
        if not src.is_file() or sha256(src) != row['image_sha256']:
            raise ValueError(f'Imagem do Drive ausente ou alterada: {rel}')
        if dst.exists():
            if sha256(dst) != row['image_sha256']:
                raise ValueError(f'Imagem local alterada: {rel}')
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            if sha256(dst) != row['image_sha256']:
                raise ValueError(f'Falha na cópia: {rel}')
        if index % 100 == 0:
            print(f'Cópia/verificação: {index}/{len(rows)}', flush=True)
    if not local_manifest.exists():
        shutil.copy2(manifest, local_manifest)
    print('AIGI local pronto:', args.local, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'stage'))
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--local', type=Path, default=LOCAL)
    parser.add_argument('--cache', type=Path, default=CACHE)
    parser.add_argument('--real-cache', type=Path, default=REAL_CACHE)
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare(args)
    else:
        stage(args)


if __name__ == '__main__':
    main()
