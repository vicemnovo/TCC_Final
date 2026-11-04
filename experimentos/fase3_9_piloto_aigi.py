"""Piloto de novos pares AIGI para a 3.9; não seleciona dados nem treina.

Baixa o CSV e os shards escolhidos por inteiro para /content antes de lê-los.
Exclui UIDs usados na 3.8 e tenta algumas fotografias reais para estimar a
disponibilidade. O relatório não contém URLs nem credenciais.
"""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import urllib.error

from fase3_8_aigi_dados import REVISION, ROOT, SHARDS, EXTRA_SHARDS, real_url
from fase3_8_aigi_piloto import REPO, REAL_CSV, download_probe, real_rows, sha256


LOCAL = Path('/content/fase3_9_aigi_piloto')
DEFAULT_GROUPS = ('FLUX.1_dev', 'PA_alpha', 'SD_2.1')


def new_candidates(parquet_paths, real, used_uids, side=256):
    """Conta UIDs novos pareáveis sem ler as imagens dentro dos Parquet."""
    import pyarrow.parquet as pq

    found = {}
    for group, paths in parquet_paths.items():
        candidates = []
        seen = set()
        for path in paths:
            parquet = pq.ParquetFile(path)
            required = {'uid', 'generator', 'height', 'width', 'image'}
            if not required <= set(parquet.schema_arrow.names):
                raise ValueError(f'Esquema AIGI incompleto: {path.name}')
            for batch in parquet.iter_batches(
                    columns=['uid', 'generator', 'height', 'width'], batch_size=512):
                for row in batch.to_pylist():
                    uid = str(row['uid'])
                    if row['generator'] != group or not re.fullmatch(r'[A-Za-z0-9_-]{8}', uid):
                        raise ValueError(f'Gerador ou UID inesperado: {path.name}')
                    if uid in seen:
                        raise ValueError(f'UID repetido em shards de {group}')
                    seen.add(uid)
                    source = real.get(uid)
                    url = real_url(source) if source else None
                    if (uid not in used_uids and url and
                            min(int(row['height']), int(row['width'])) >= side):
                        candidates.append({'uid': uid, 'source_dataset': source['source_dataset'],
                                           'image_path': url})
        candidates.sort(key=lambda row: hashlib.sha256(
            f"tcc-fase3-9-piloto:{group}:{row['uid']}".encode()).hexdigest())
        found[group] = candidates
        print(f'{group}: {len(candidates)} UIDs novos pareáveis em '
              f'{len(paths)} shard(s); {len(seen)} UIDs examinados', flush=True)
    return found


def pilot(args):
    if not args.root.is_dir():
        raise FileNotFoundError('Monte o Drive antes do piloto')
    previous_path = args.root / 'dados_fase3_8/aigi_pares_256_v1/manifest.json'
    if not previous_path.is_file():
        raise FileNotFoundError(previous_path)
    previous = json.loads(previous_path.read_text(encoding='utf-8'))
    if previous.get('revision') != REVISION or previous.get('name') != 'aigi_pares_256_v1':
        raise ValueError('Manifesto da 3.8 não corresponde ao dataset fixado')
    used_uids = {row['uid'] for row in previous['rows']}
    if len(used_uids) * 2 != len(previous['rows']):
        raise ValueError('Pares anteriores incompletos; não selecionar novos UIDs')
    if shutil.disk_usage(args.local.parent).free < 3_000_000_000:
        raise RuntimeError('/content com menos de 3 GB livres para o piloto')
    args.local.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import hf_hub_download

    groups = args.groups
    unknown = set(groups) - set(SHARDS)
    if unknown or len(groups) != len(set(groups)):
        raise ValueError(f'Geradores inválidos/repetidos: {sorted(unknown)}')
    names = {REAL_CSV}
    for group in groups:
        names.add(SHARDS[group])
        if group in EXTRA_SHARDS:
            names.add(EXTRA_SHARDS[group])
    files = {}
    for name in sorted(names):
        print('Baixando/verificando arquivo completo:', name, flush=True)
        files[name] = Path(hf_hub_download(
            repo_id=REPO, repo_type='dataset', filename=name, revision=REVISION,
            local_dir=args.local / 'source', token=os.environ.get('HF_TOKEN') or False))
        print(f'  pronto: {files[name].stat().st_size / 1e6:.1f} MB', flush=True)
    real, _ = real_rows(files[REAL_CSV])
    paths = {group: [files[SHARDS[group]]] +
             ([files[EXTRA_SHARDS[group]]] if group in EXTRA_SHARDS else [])
             for group in groups}
    candidates = new_candidates(paths, real, used_uids)
    report = {'name': 'fase3_9_aigi_piloto_v1', 'revision': REVISION,
              'previous_manifest_sha256': sha256(previous_path),
              'source_sha256': {name: sha256(files[name]) for name in sorted(files)},
              'groups': {}, 'training_started': False, 'partitions_created': False}
    for group in groups:
        success, failures, sizes, sources = 0, Counter(), Counter(), Counter()
        for index, row in enumerate(candidates[group][:args.probes], 1):
            try:
                result = download_probe({'uid': row['uid'], **row}, args.local)
                sizes['256_plus' if min(result['size']) >= 256 else 'under_256'] += 1
                if min(result['size']) >= 256:
                    success += 1
                    sources[row['source_dataset']] += 1
                print(f'{group}: foto real {index}/{min(args.probes,len(candidates[group]))} '
                      f'legível; lado curto={min(result["size"])}', flush=True)
            except (OSError, ValueError, urllib.error.URLError) as error:
                failures[type(error).__name__] += 1
                print(f'{group}: foto real {index}/{min(args.probes,len(candidates[group]))} '
                      f'indisponível ({type(error).__name__})', flush=True)
        report['groups'][group] = {
            'new_paired_https_candidates': len(candidates[group]),
            'probes_attempted': min(args.probes, len(candidates[group])),
            'probes_usable': success, 'probe_size': dict(sizes),
            'probe_sources': dict(sources), 'probe_failures': dict(failures)}
    output = args.local / 'pilot_summary.json'
    if output.is_file():
        if json.loads(output.read_text(encoding='utf-8')) != report:
            raise ValueError('Relatório anterior difere; preserve-o antes de outro piloto')
    else:
        with output.open('x', encoding='utf-8') as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print('Piloto concluído; nenhum dado de treino ou checkpoint foi criado.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--local', type=Path, default=LOCAL)
    parser.add_argument('--groups', nargs='+', default=list(DEFAULT_GROUPS))
    parser.add_argument('--probes', type=int, default=5)
    args = parser.parse_args()
    if not 1 <= args.probes <= 20:
        raise ValueError('Escolha 1 a 20 provas por gerador')
    pilot(args)


if __name__ == '__main__':
    main()
