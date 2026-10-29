"""Piloto, preparo imutável e cópia local de dados internos da Fase 3.5.

Baixa arquivos Parquet completos para /content; nunca usa streaming remoto.
Não lê predições, métricas ou resultados do benchmark externo.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import io
import json
import os
from pathlib import Path
import shutil

ROOT = Path('/content/drive/MyDrive/TCC_Final')
LOCAL = Path('/content/fase3_5_novos_v1')
NAME = 'fase3_5_novos_v1'
REPO = 'OwensLab/CommunityForensics-Small'
LICENSE = 'CC-BY-NC-SA-4.0; somente pesquisa não comercial'
SEED = 3505
MIN_GROUP = 20
SPLITS = ('train', 'validation_selection', 'validation_calibration')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save_exclusive(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)


def pinned_repo():
    from huggingface_hub import HfApi
    info = HfApi(token=False).dataset_info(REPO)
    if not info.sha or len(info.sha) != 40:
        raise RuntimeError('Não foi possível fixar o commit do dataset')
    return info.sha


def download_shard(shard, revision, cache_dir):
    from huggingface_hub import hf_hub_download
    if not shard.startswith('data/HFCF_small_') or not shard.endswith('.parquet') or '..' in shard:
        raise ValueError(f'Shard inesperado: {shard}')
    return Path(hf_hub_download(repo_id=REPO, repo_type='dataset', filename=shard,
                                revision=revision, local_dir=cache_dir, token=False))


def metadata_rows(path):
    import pyarrow.parquet as pq
    columns = ('label', 'model_name', 'real_source', 'split', 'nsfw_flag', 'image_name')
    parquet = pq.ParquetFile(path)
    if not set(columns + ('image_data',)) <= set(parquet.schema.names):
        raise ValueError(f'Colunas esperadas ausentes em {path.name}')
    index = 0
    for batch in parquet.iter_batches(columns=list(columns), batch_size=512):
        for row in batch.to_pylist():
            yield index, row
            index += 1


def group_of(row):
    label = int(row['label'])
    if label not in (0, 1):
        raise ValueError('Rótulo fora de real=0/fake=1')
    raw = row['model_name'] if label else row['real_source']
    group = str(raw or '').strip().lower()
    if not group or group in ('none', 'nan', 'unknown'):
        return None
    return label, group


def candidate_rows(paths):
    groups = defaultdict(list)
    for shard, path in paths.items():
        count = Counter()
        for index, row in metadata_rows(path):
            if str(row.get('split', '')).lower() != 'train' or row.get('nsfw_flag') is True:
                continue
            key = group_of(row)
            if key is None:
                continue
            groups[key].append({'shard': shard, 'index': index, 'label': key[0],
                                'group': key[1], 'image_name': str(row['image_name'])})
            count[key[0]] += 1
        print(f'{shard}: candidatos reais={count[0]} sintéticos={count[1]}', flush=True)
    return groups


def priority(value):
    return hashlib.sha256(f'{SEED}:{value}'.encode()).hexdigest()


def select_rows(groups, per_class, per_group):
    selected = []
    for label in (0, 1):
        eligible = {group: sorted(rows, key=lambda row: priority(
            f"{row['shard']}:{row['index']}"))[:per_group]
                    for (lab, group), rows in groups.items()
                    if lab == label and len(rows) >= MIN_GROUP}
        if len(eligible) < 3:
            raise ValueError(f'Classe {label}: menos de três grupos com {MIN_GROUP}+ imagens; '
                             'escolha outros shards antes de preparar')
        order = sorted(eligible, key=priority)
        picks = []
        while len(picks) < per_class and any(eligible.values()):
            for group in order:
                if eligible[group] and len(picks) < per_class:
                    picks.append(eligible[group].pop())
        if len(picks) != per_class:
            raise ValueError(f'Classe {label}: {len(picks)}/{per_class} candidatas elegíveis')
        chosen_groups = sorted({row['group'] for row in picks}, key=priority)
        if len(chosen_groups) < 3:
            raise ValueError('Não foi possível reservar grupos distintos por divisão')
        assignment = {chosen_groups[0]: 'validation_selection',
                      chosen_groups[1]: 'validation_calibration'}
        # Com muitas fontes, reserva aproximadamente um quinto dos grupos em
        # cada validação; nunca divide imagens do mesmo gerador/fonte.
        for index, group in enumerate(chosen_groups[2:], 2):
            if index % 5 == 3:
                assignment[group] = 'validation_selection'
            elif index % 5 == 4:
                assignment[group] = 'validation_calibration'
            else:
                assignment[group] = 'train'
        for row in picks:
            selected.append({**row, 'subset': assignment[row['group']]})
    counts = Counter((row['subset'], row['label']) for row in selected)
    if any(counts[subset, label] < MIN_GROUP for subset in SPLITS for label in (0, 1)):
        raise ValueError(f'Uma divisão ficou pequena demais: {dict(counts)}')
    return sorted(selected, key=lambda row: (row['subset'], row['label'], row['group'],
                                            row['shard'], row['index']))


def known_hashes(root):
    known = set()
    benchmark = root / 'dados_externos/benchmark_externo_raise_mnw_2000_v1/manifest.json'
    if not benchmark.is_file():
        raise FileNotFoundError(f'Benchmark externo congelado não encontrado: {benchmark}')
    for row in json.loads(benchmark.read_text(encoding='utf-8'))['rows']:
        known.update((row['original_sha256'], row['image_sha256']))
    old = root / 'dados_fase3_4/manifest.json'
    if not old.is_file():
        raise FileNotFoundError(old)
    for subset in ('train', 'test'):
        rows = json.loads(old.read_text(encoding='utf-8'))[subset]
        for index, row in enumerate(rows, 1):
            path = Path(row['path'])
            if not path.is_file():
                raise FileNotFoundError(path)
            known.add(sha(path))
            if index % 500 == 0:
                print(f'Hashes OpenFake {subset}: {index}/{len(rows)}', flush=True)
    for folder in (root / 'minhas_imagens', root / 'teste_externo',
                   root / 'validacao_fase3/teste_externo'):
        if folder.is_dir():
            for path in folder.rglob('*'):
                if path.is_file() and path.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp'):
                    known.add(sha(path))
    return known


def encode_jpeg(raw):
    from PIL import Image, ImageOps
    with Image.open(io.BytesIO(raw)) as opened:
        image = ImageOps.exif_transpose(opened).convert('RGB')
    image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
    if min(image.size) < 224:
        scale = 224 / min(image.size)
        image = image.resize((round(image.width * scale), round(image.height * scale)),
                             Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', quality=90, subsampling=0)
    return buffer.getvalue(), image.size


def prepare(args):
    import pyarrow.parquet as pq
    root = args.root
    if shutil.disk_usage(root).free < 2_000_000_000:
        raise RuntimeError('Drive com menos de 2 GB livres; não iniciar preparo')
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(args.cache_dir).free < 2_000_000_000:
        raise RuntimeError('/content com menos de 2 GB livres; não baixar shard')
    final = root / 'dados_fase3_5' / NAME
    work = root / 'dados_fase3_5' / (NAME + '.incompleto')
    if final.exists():
        raise FileExistsError(f'Dados já congelados; não alterar: {final}')
    work.parent.mkdir(parents=True, exist_ok=True)
    plan_path = work / 'plan.json'
    if plan_path.is_file():
        plan = json.loads(plan_path.read_text(encoding='utf-8'))
        if (plan['shards'] != args.shards or plan['per_class'] != args.per_class
                or plan['per_group'] != args.per_group):
            raise ValueError('Parâmetros diferem do plano persistido; não trocar seleção')
        print('Retomando plano existente:', plan_path, flush=True)
    elif work.exists():
        raise FileExistsError(f'Pasta incompleta sem plano: {work}')
    else:
        revision = pinned_repo()
        paths = {shard: download_shard(shard, revision, args.cache_dir)
                 for shard in args.shards}
        groups = candidate_rows(paths)
        rows = select_rows(groups, args.per_class, args.per_group)
        plan = {'name': NAME, 'source': REPO, 'license': LICENSE, 'revision': revision,
                'shards': args.shards, 'shard_sha256': {key: sha(path) for key, path in paths.items()},
                'per_class': args.per_class, 'per_group': args.per_group,
                'seed': SEED, 'rows': rows}
        work.mkdir(exist_ok=False)
        save_exclusive(plan_path, plan)
        print('Seleção persistida:', plan_path, flush=True)
    paths = {shard: download_shard(shard, plan['revision'], args.cache_dir)
             for shard in plan['shards']}
    for shard, path in paths.items():
        if sha(path) != plan['shard_sha256'][shard]:
            raise ValueError(f'Shard mudou: {shard}')
    forbidden = known_hashes(root)
    records_path = work / 'records.json'
    records = json.loads(records_path.read_text(encoding='utf-8')) if records_path.exists() else {}
    wanted = {(row['shard'], row['index']): row for row in plan['rows']}
    selected_raw_hashes = set()
    selected_image_hashes = set()
    for shard, path in paths.items():
        parquet = pq.ParquetFile(path)
        index = 0
        for batch in parquet.iter_batches(columns=['image_data'], batch_size=32):
            for item in batch.to_pylist():
                row = wanted.get((shard, index))
                index += 1
                if row is None:
                    continue
                key = f"{shard}:{row['index']}"
                raw = item['image_data']
                if not isinstance(raw, bytes):
                    raise ValueError(f'Imagem inválida: {key}')
                raw_hash = hashlib.sha256(raw).hexdigest()
                if key in records:
                    prior = records[key]
                    image_path = work / prior['image']
                    if prior['original_sha256'] != raw_hash or not image_path.is_file() or sha(image_path) != prior['image_sha256']:
                        raise ValueError(f'Retomada encontrou alteração: {key}')
                    image_hash = prior['image_sha256']
                else:
                    jpeg, size = encode_jpeg(raw)
                    image_hash = hashlib.sha256(jpeg).hexdigest()
                    rel = f"images/{row['label']}/{image_hash}.jpg"
                    image_path = work / rel
                    if image_path.exists():
                        if sha(image_path) != image_hash:
                            raise ValueError(f'JPEG parcial ou alterado: {key}')
                    else:
                        image_path.parent.mkdir(parents=True, exist_ok=True)
                        with image_path.open('xb') as stream:
                            stream.write(jpeg)
                    records[key] = {**row, 'image': rel, 'original_sha256': raw_hash,
                                    'image_sha256': image_hash, 'size': size}
                    temp = records_path.with_suffix('.json.tmp')
                    temp.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding='utf-8')
                    os.replace(temp, records_path)
                if raw_hash in forbidden or image_hash in forbidden:
                    raise ValueError(f'Sobreposição exata com conjunto reservado/histórico: {key}')
                if raw_hash in selected_raw_hashes or image_hash in selected_image_hashes:
                    raise ValueError(f'Duplicata interna: {key}')
                selected_raw_hashes.add(raw_hash)
                selected_image_hashes.add(image_hash)
                if len(selected_raw_hashes) % 100 == 0:
                    print(f'Preparação: {len(selected_raw_hashes)}/{len(wanted)}', flush=True)
    if len(records) != len(wanted) or len(selected_raw_hashes) != len(wanted):
        raise ValueError('Seleção incompleta ou duplicada; não congelar')
    manifest = {'name': NAME, 'source': REPO, 'license': LICENSE,
                'revision': plan['revision'], 'plan_sha256': sha(plan_path),
                'rows': [records[f"{row['shard']}:{row['index']}"] for row in plan['rows']]}
    manifest_path = work / 'manifest.json'
    if manifest_path.is_file():
        if json.loads(manifest_path.read_text(encoding='utf-8')) != manifest:
            raise ValueError('Manifesto parcial difere da seleção registrada')
    else:
        save_exclusive(manifest_path, manifest)
    work.rename(final)
    print('Dados internos congelados:', final, flush=True)


def stage(args):
    source = args.root / 'dados_fase3_5' / NAME
    manifest = source / 'manifest.json'
    if not manifest.is_file():
        raise FileNotFoundError(f'Execute prepare antes: {manifest}')
    rows = json.loads(manifest.read_text(encoding='utf-8'))['rows']
    args.local_dir.mkdir(parents=True, exist_ok=True)
    local_manifest = args.local_dir / 'manifest.json'
    if local_manifest.exists() and sha(local_manifest) != sha(manifest):
        raise ValueError('Manifesto local de outra seleção; não sobrescrever')
    for index, row in enumerate(rows, 1):
        rel = Path(row['image'])
        if rel.is_absolute() or '..' in rel.parts:
            raise ValueError('Caminho inseguro no manifesto')
        src, dst = source / rel, args.local_dir / rel
        if not src.is_file() or sha(src) != row['image_sha256']:
            raise ValueError(f'Imagem no Drive alterada: {rel}')
        if dst.exists():
            if sha(dst) != row['image_sha256']:
                raise ValueError(f'Imagem local alterada: {rel}')
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            if sha(dst) != row['image_sha256']:
                raise ValueError(f'Cópia local falhou: {rel}')
        if index % 100 == 0 or index == len(rows):
            print(f'Cópia/verificação: {index}/{len(rows)}', flush=True)
    if not local_manifest.exists():
        shutil.copy2(manifest, local_manifest)
    print('Dados internos locais prontos:', args.local_dir, flush=True)


def pilot(args):
    revision = pinned_repo()
    paths = {shard: download_shard(shard, revision, args.cache_dir)
             for shard in args.shards}
    groups = candidate_rows(paths)
    counts = Counter({label: sum(len(rows) for (lab, _), rows in groups.items()
                                 if lab == label) for label in (0, 1)})
    print('Piloto:', dict(counts), '| grupos reais=', sum(lab == 0 for lab, _ in groups),
          '| geradores=', sum(lab == 1 for lab, _ in groups),
          '| bytes=', sum(path.stat().st_size for path in paths.values()),
          '| commit=', revision, flush=True)
    for label in (0, 1):
        eligible = sorted(((group, len(rows)) for (lab, group), rows in groups.items()
                           if lab == label and len(rows) >= MIN_GROUP),
                          key=lambda item: (-item[1], item[0]))
        print(f'Classe {label}: grupos elegíveis (20+): {len(eligible)}; '
              f'principais: {eligible[:12]}', flush=True)
    print('Piloto não cria seleção nem inicia treino.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('pilot', 'prepare', 'stage'))
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--local-dir', type=Path, default=LOCAL)
    parser.add_argument('--cache-dir', type=Path, default=Path('/content/fase3_5_parquet'))
    parser.add_argument('--shard', action='append', dest='shards')
    parser.add_argument('--per-class', type=int, default=1500)
    parser.add_argument('--per-group', type=int, default=750)
    args = parser.parse_args()
    args.shards = args.shards or ['data/HFCF_small_117.parquet']
    if args.per_class < 100 or args.per_group < MIN_GROUP or args.per_class < args.per_group:
        raise ValueError('Limites de seleção inválidos')
    if args.mode == 'pilot':
        pilot(args)
    elif args.mode == 'prepare':
        prepare(args)
    else:
        stage(args)


if __name__ == '__main__':
    main()
