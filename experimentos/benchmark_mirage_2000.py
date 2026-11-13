"""Congela e avalia 2.000 imagens externas Mirage-Test com a Fase 3.5.

`pilot`, `prepare`, `stage` e `evaluate` são ações manuais separadas. Nenhuma
delas treina, calibra ou escolhe hiperparâmetros com o conjunto externo.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath


ROOT = Path('/content/drive/MyDrive/TCC_Final')
NAME = 'benchmark_externo_mirage_2000_v1'
REPO = 'Yunncheng/Mirage-Test'
DOMAINS = ('Animal', 'Human', 'Object', 'Scene')
PER_DOMAIN_CLASS = 250
TOTAL = 2 * len(DOMAINS) * PER_DOMAIN_CLASS
SEED = 'TCC-Mirage-2000-v1-2026-10-29'
RUN = 'fase3_5_cfsmall_v1'
EXPECTED_CHECKPOINT_SHA256 = '1dd78d4cd32106d1471d388fa3a96a094c3906e7860ed4ad5ec3d7acad9b9045'
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save_exclusive(path, value):
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def save_atomic(path, value):
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(tmp, path)


def safe_source_path(value):
    if not isinstance(value, str) or '\\' in value:
        raise ValueError('Caminho de origem inválido')
    path = PurePosixPath(value)
    if (path.is_absolute() or len(path.parts) < 3 or '..' in path.parts
            or path.parts[0] not in DOMAINS
            or path.parts[1] not in ('0_real', '1_fake')
            or path.suffix.lower() not in IMAGE_EXTENSIONS):
        raise ValueError(f'Caminho de origem fora das quatro categorias fotográficas: {value!r}')
    return path


def priority(value):
    return hashlib.sha256(f'{SEED}|{value}'.encode()).hexdigest()


def hub_token():
    """Usa apenas a variável de sessão, se fornecida pelo notebook."""
    return os.environ.get('HF_TOKEN') or False


def select_rows(metadata):
    """Seleciona antes do download; não olha pixels nem respostas do modelo."""
    buckets = defaultdict(list)
    for item in metadata:
        raw = str(item['file_name'])
        path = PurePosixPath(raw)
        if path.parts and path.parts[0] == 'Anime':
            continue  # Anime só contém a classe sintética no cartão oficial.
        path = safe_source_path(raw)
        label = 0 if path.parts[1] == '0_real' else 1
        if item['is_real'] != path.parts[1]:
            raise ValueError(f'Rótulo do índice difere da pasta: {raw}')
        if item['content_type'] != path.parts[0]:
            raise ValueError(f'Domínio do índice difere da pasta: {raw}')
        buckets[path.parts[0], label].append(raw)
    result = []
    for domain in DOMAINS:
        for label in (0, 1):
            options = sorted(set(buckets[domain, label]), key=priority)
            if len(options) < PER_DOMAIN_CLASS:
                raise ValueError(f'{domain}, classe {label}: apenas {len(options)} imagens')
            if label == 1:
                by_generator = defaultdict(list)
                for name in options:
                    path = PurePosixPath(name)
                    by_generator['/'.join(path.parts[2:-1]) or 'nao_informado'].append(name)
                # Distribui a seleção entre geradores declarados sem consultar
                # predições. Cada grupo contribui até a capacidade disponível.
                generators = sorted(by_generator, key=priority)
                chosen = []
                while len(chosen) < PER_DOMAIN_CLASS:
                    advanced = False
                    for generator in generators:
                        if by_generator[generator] and len(chosen) < PER_DOMAIN_CLASS:
                            chosen.append(by_generator[generator].pop(0))
                            advanced = True
                    if not advanced:
                        raise ValueError(f'{domain}: candidatos sintéticos insuficientes')
            else:
                chosen = options[:PER_DOMAIN_CLASS]
            for source_file in chosen:
                path = PurePosixPath(source_file)
                result.append({'source_file': source_file, 'label': label,
                               'domain': domain,
                               'group': '/'.join(path.parts[2:-1]) or 'nao_informado',
                               'image': 'images/' + source_file})
    if len(result) != TOTAL or len({r['image'] for r in result}) != TOTAL:
        raise ValueError('Seleção externa incompleta ou com caminhos repetidos')
    return result


def source_index(cache_dir, revision=None):
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    revision = revision or HfApi(token=hub_token()).dataset_info(REPO).sha
    if not isinstance(revision, str) or len(revision) != 40:
        raise ValueError('Commit do Mirage-Test inválido')
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = Path(hf_hub_download(repo_id=REPO, repo_type='dataset',
                                filename='test.parquet', revision=revision,
                                local_dir=cache_dir, token=hub_token()))
    table = pq.read_table(path, columns=['file_name', 'is_real', 'content_type'])
    rows = select_rows(table.to_pylist())
    print(f'Índice Mirage-Test: {table.num_rows} linhas | seleção fixa: {len(rows)} | '
          f'commit: {revision}', flush=True)
    print('Domínios/classes:', dict(Counter((r['domain'], r['label']) for r in rows)), flush=True)
    return revision, digest(path), rows


def frozen_model(root, code_root):
    run = root / 'modelos' / RUN
    checkpoint = run / 'phase3_final.pt'
    config_path = run / 'experiment_config.json'
    metrics_path = run / 'internal_metrics.json'
    complete_path = run / 'training_complete.json'
    source = code_root / 'Notebooks/fase_3/fase3_3/fase3.3_compressao.ipynb'
    for path in (checkpoint, config_path, metrics_path, complete_path, source):
        if not path.is_file():
            raise FileNotFoundError(path)
    if digest(checkpoint) != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError('Checkpoint da Fase 3.5 difere do hash do diagnóstico interno')
    config = json.loads(config_path.read_text(encoding='utf-8'))
    metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
    complete = json.loads(complete_path.read_text(encoding='utf-8'))
    if (complete.get('status') != 'internal_training_complete'
            or config.get('RUN_NAME') != RUN
            or config.get('SOURCE_SHA256') != digest(source)):
        raise ValueError('Execução 3.5 ou arquitetura não corresponde ao treino')
    # O próprio checkpoint contém os valores exatos. A conferência não reestima nada.
    import torch
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    t, threshold = float(saved['temperature']), float(saved['threshold'])
    if (saved.get('cfg_phase3', {}).get('RUN_NAME') != RUN
            or abs(t - float(metrics['temperature'])) > 1e-8
            or abs(threshold - float(metrics['threshold'])) > 1e-8
            or not 0 < t < 10 or not 0 < threshold < 1):
        raise ValueError('Checkpoint e métricas internas divergem')
    return {'checkpoint_sha256': digest(checkpoint),
            'config_sha256': digest(config_path), 'architecture_sha256': digest(source),
            'temperature': t, 'threshold': threshold,
            'protocol': 'five_crop_224; upscale_bicubic_only_if_short_side_below_224; mean_logits'}


def pilot(args):
    revision, index_sha, rows = source_index(args.cache_dir)
    print('Piloto apenas de metadados; não baixa as 2.000 imagens nem cria seleção no Drive.',
          '| índice SHA-256:', index_sha, '| primeira imagem:', rows[0]['source_file'], flush=True)


def known_hashes(root):
    """Compara conteúdo com treino, teste histórico e avaliações anteriores."""
    known = set()
    cf = root / 'dados_fase3_5/fase3_5_novos_v1/manifest.json'
    old = root / 'dados_fase3_4/manifest.json'
    prior = root / 'dados_externos/benchmark_externo_raise_mnw_2000_v1/manifest.json'
    for path in (cf, old, prior):
        if not path.is_file():
            raise FileNotFoundError(path)
    for row in json.loads(cf.read_text(encoding='utf-8'))['rows']:
        known.update((row['original_sha256'], row['image_sha256']))
    for row in json.loads(prior.read_text(encoding='utf-8'))['rows']:
        known.update((row['original_sha256'], row['image_sha256']))
    old_data = json.loads(old.read_text(encoding='utf-8'))
    for subset in ('train', 'test'):
        for index, row in enumerate(old_data[subset], 1):
            path = Path(row['path'])
            if not path.is_file():
                raise FileNotFoundError(path)
            known.add(digest(path))
            if index % 500 == 0:
                print(f'Sobreposição: OpenFake {subset} {index}/{len(old_data[subset])}', flush=True)
    for folder in (root / 'minhas_imagens', root / 'validacao_fase3/teste_externo'):
        if folder.is_dir():
            for path in folder.rglob('*'):
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                    known.add(digest(path))
    return known


def verify_image(path):
    from PIL import Image
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        return [image.width, image.height], image.format


def prepare(args):
    final = args.root / 'dados_externos' / NAME
    work = args.root / 'dados_externos' / (NAME + '.incompleto')
    if final.exists():
        raise FileExistsError(f'Benchmark congelado já existe: {final}')
    if shutil.disk_usage(args.root).free < 3_000_000_000:
        raise RuntimeError('Drive com menos de 3 GB livres; interrompendo preparo')
    if shutil.disk_usage(args.cache_dir.parent).free < 4_000_000_000:
        raise RuntimeError('/content com menos de 4 GB livres; interrompendo preparo')
    plan_path = work / 'plan.json'
    if plan_path.is_file():
        plan = json.loads(plan_path.read_text(encoding='utf-8'))
        if plan.get('name') != NAME or len(plan.get('rows', [])) != TOTAL:
            raise ValueError('Plano de retomada inválido')
        revision, index_sha, rows = source_index(args.cache_dir, plan['revision'])
        if index_sha != plan['index_sha256'] or rows != plan['rows']:
            raise ValueError('Índice ou seleção da retomada mudou')
        print('Retomando plano já registrado:', plan_path, flush=True)
    elif work.exists():
        raise FileExistsError(f'Pasta incompleta sem plano: {work}')
    else:
        revision, index_sha, rows = source_index(args.cache_dir)
        model = frozen_model(args.root, args.code_root)
        plan = {'name': NAME, 'repo': REPO, 'license': 'MIT conforme cartão do dataset',
                'revision': revision, 'index_sha256': index_sha, 'seed': SEED,
                'per_domain_class': PER_DOMAIN_CLASS, 'model': model,
                'rows': rows}
        work.mkdir(parents=True, exist_ok=False)
        save_exclusive(plan_path, plan)
        print('Plano fixado antes das imagens:', plan_path, flush=True)
    if frozen_model(args.root, args.code_root) != plan['model']:
        raise ValueError('Modelo congelado difere do plano')
    forbidden = known_hashes(args.root)
    records_path = work / 'records.json'
    records = json.loads(records_path.read_text(encoding='utf-8')) if records_path.is_file() else {}
    if set(records) - {r['source_file'] for r in plan['rows']}:
        raise ValueError('Registros extras na preparação')
    from huggingface_hub import hf_hub_download
    seen = set()
    for index, row in enumerate(plan['rows'], 1):
        relative = safe_source_path(row['source_file'])
        local = Path(hf_hub_download(repo_id=REPO, repo_type='dataset',
                                     filename=str(relative), revision=revision,
                                     local_dir=args.cache_dir, token=hub_token()))
        image_sha = digest(local)
        if image_sha in seen or image_sha in forbidden:
            raise ValueError(f'Duplicata exata interna ou sobreposição histórica: {relative}')
        seen.add(image_sha)
        size, fmt = verify_image(local)
        dest = work / row['image']
        record = records.get(row['source_file'])
        if record is not None and record.get('image_sha256') != image_sha:
            raise ValueError(f'Imagem da fonte mudou na retomada: {relative}')
        if dest.exists():
            if digest(dest) != image_sha:
                raise ValueError(f'Cópia do Drive alterada: {dest}')
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            temp = Path(str(dest) + '.part')
            if temp.exists() and digest(temp) != image_sha:
                raise ValueError(f'Cópia parcial diferente: {temp}')
            if not temp.exists():
                shutil.copyfile(local, temp)
            if digest(temp) != image_sha:
                raise ValueError(f'Falha na cópia ao Drive: {temp}')
            temp.rename(dest)
        expected = {**row, 'image_sha256': image_sha, 'size': size, 'format': fmt}
        if record is not None and record != expected:
            raise ValueError(f'Registro mudou: {relative}')
        if record is None:
            records[row['source_file']] = expected
            save_atomic(records_path, records)
        if index % 50 == 0 or index == TOTAL:
            print(f'Preparação Mirage-Test: {index}/{TOTAL}', flush=True)
    if len(records) != TOTAL or len(seen) != TOTAL:
        raise ValueError('Benchmark externo incompleto')
    manifest = {'name': NAME, 'repo': REPO, 'revision': revision,
                'plan_sha256': digest(plan_path), 'model': plan['model'],
                'exact_overlap_checked': True,
                'rows': [records[row['source_file']] for row in plan['rows']]}
    manifest_path = work / 'manifest.json'
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding='utf-8')) != manifest:
            raise ValueError('Manifesto incompleto difere do plano; não sobrescrever')
    else:
        save_exclusive(manifest_path, manifest)
    work.rename(final)
    print('Benchmark externo congelado:', final, flush=True)


def stage(args):
    source = args.root / 'dados_externos' / NAME
    manifest = source / 'manifest.json'
    if not manifest.is_file():
        raise FileNotFoundError(f'Execute prepare antes: {manifest}')
    data = json.loads(manifest.read_text(encoding='utf-8'))
    rows = data['rows']
    if len(rows) != TOTAL or Counter(r['label'] for r in rows) != {0: TOTAL // 2, 1: TOTAL // 2}:
        raise ValueError('Contagem de classes do manifesto alterada')
    if shutil.disk_usage(args.local_dir.parent).free < 3_000_000_000:
        raise RuntimeError('/content com menos de 3 GB livres para copiar o benchmark')
    args.local_dir.mkdir(parents=True, exist_ok=True)
    local_manifest = args.local_dir / 'manifest.json'
    if local_manifest.exists() and digest(local_manifest) != digest(manifest):
        raise ValueError('Manifesto local de outra seleção')
    for index, row in enumerate(rows, 1):
        rel = PurePosixPath(row['image'])
        if rel.is_absolute() or '..' in rel.parts or rel.parts[0] != 'images':
            raise ValueError('Caminho inseguro no manifesto')
        src, dst = source / str(rel), args.local_dir / str(rel)
        if not src.is_file() or digest(src) != row['image_sha256']:
            raise ValueError(f'Imagem do Drive ausente ou alterada: {rel}')
        if dst.exists():
            if digest(dst) != row['image_sha256']:
                raise ValueError(f'Imagem local alterada: {rel}')
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
            if digest(dst) != row['image_sha256']:
                raise ValueError(f'Cópia local falhou: {rel}')
        if index % 100 == 0 or index == TOTAL:
            print(f'Cópia/verificação: {index}/{TOTAL}', flush=True)
    if not local_manifest.exists():
        shutil.copyfile(manifest, local_manifest)
    print('Benchmark local pronto:', args.local_dir, flush=True)


def wilson(correct, total):
    z = 1.959963984540054
    p = correct / total
    den = 1 + z*z/total
    mid = (p + z*z/(2*total)) / den
    half = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / den
    return [mid-half, mid+half]


def evaluate(args):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image, ImageOps
    from sklearn.metrics import roc_auc_score
    from torchvision import transforms
    from avaliar_benchmark_2000 import load_architecture, load_model

    output = args.root / 'resultados' / NAME / RUN
    if output.exists():
        raise FileExistsError(f'Resultado anterior preservado: {output}')
    manifest = args.local_dir / 'manifest.json'
    if not manifest.is_file():
        raise FileNotFoundError('Execute stage antes da avaliação')
    data = json.loads(manifest.read_text(encoding='utf-8'))
    source_manifest = args.root / 'dados_externos' / NAME / 'manifest.json'
    if not source_manifest.is_file() or digest(manifest) != digest(source_manifest):
        raise ValueError('Manifesto local difere do Drive')
    rows = data['rows']
    if len(rows) != TOTAL or Counter(r['label'] for r in rows) != {0: TOTAL // 2, 1: TOTAL // 2}:
        raise ValueError('Contagem externa alterada')
    frozen = frozen_model(args.root, args.code_root)
    if data.get('model') != frozen or not data.get('exact_overlap_checked'):
        raise ValueError('Modelo ou auditoria de sobreposição difere do congelamento')
    for index, row in enumerate(rows, 1):
        rel = PurePosixPath(row['image'])
        if rel.is_absolute() or '..' in rel.parts or rel.parts[0] != 'images':
            raise ValueError('Caminho inseguro no manifesto')
        if digest(args.local_dir / str(rel)) != row['image_sha256']:
            raise ValueError(f'Imagem externa local alterada: {rel}')
        if index % 100 == 0:
            print(f'Integridade antes da inferência: {index}/{TOTAL}', flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError('Conecte uma GPU no Colab para avaliar')
    device = torch.device('cuda')
    phase1_cls, phase2_cls, arch_sha = load_architecture(args.code_root, device)
    checkpoint = args.root / 'modelos' / RUN / 'phase3_final.pt'
    model, temperature, threshold, _ = load_model(
        RUN, checkpoint, phase1_cls, phase2_cls, device)
    if temperature != frozen['temperature'] or threshold != frozen['threshold']:
        raise ValueError('Calibração difere da versão congelada')
    print(f'Fase 3.5 congelada | GPU={torch.cuda.get_device_name(0)} | '
          f'T={temperature:.6f} | limiar={threshold:.6f}', flush=True)
    norm = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize((0.5,)*3, (0.5,)*3)])
    crop = transforms.FiveCrop(224)
    probabilities = []
    upscaled = 0
    with torch.inference_mode():
        for start in range(0, TOTAL, 4):
            block = rows[start:start+4]
            batch = []
            for row in block:
                with Image.open(args.local_dir / row['image']) as opened:
                    image = ImageOps.exif_transpose(opened).convert('RGB')
                if min(image.size) < 224:
                    scale = 224 / min(image.size)
                    image = image.resize((round(image.width*scale), round(image.height*scale)),
                                         Image.Resampling.BICUBIC)
                    upscaled += 1
                batch.extend(norm(patch) for patch in crop(image))
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                logits = model(torch.stack(batch).to(device)).float()
            if logits.shape != (len(block)*5, 2) or not torch.isfinite(logits).all():
                raise ValueError('Logits não finitos ou forma incorreta')
            logits = logits.view(len(block), 5, 2).mean(1)
            probabilities.extend(F.softmax(logits/temperature, 1)[:, 1].cpu().tolist())
            count = start + len(block)
            if count % 100 == 0 or count == TOTAL:
                print(f'Inferência externa: {count}/{TOTAL}', flush=True)
    y = np.array([row['label'] for row in rows])
    p = np.array(probabilities)
    pred = (p >= threshold).astype(int)
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tp = int(((y == 1) & (pred == 1)).sum())
    precision = tp/(tp+fp) if tp+fp else 0.0
    recall = tp/(tp+fn)
    specificity = tn/(tn+fp)
    summary = {'model': RUN, 'n': TOTAL, 'real': TOTAL//2, 'fake': TOTAL//2,
               'accuracy': (tn+tp)/TOTAL, 'accuracy_ci95_wilson': wilson(tn+tp, TOTAL),
               'balanced_accuracy': (recall+specificity)/2,
               'auc': float(roc_auc_score(y, p)), 'precision': precision,
               'recall_fake': recall, 'specificity_real': specificity,
               'f1': 2*precision*recall/(precision+recall) if precision+recall else 0.0,
               'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
               'temperature_frozen': temperature, 'threshold_frozen': threshold,
               'upscaled_short_side_below_224': upscaled,
               'checkpoint_sha256': frozen['checkpoint_sha256'],
               'manifest_sha256': digest(manifest),
               'evaluation_script_sha256': digest(Path(__file__)),
               'architecture_source_sha256': frozen['architecture_sha256'],
               'architecture_extraction_sha256': arch_sha,
               'protocol': frozen['protocol'],
               'independence_note': 'Sem sobreposição exata com dados conhecidos; origem individual '
                                    'não é comprovada por hash e semelhança visual não foi excluída.'}
    grouped = defaultdict(list)
    for row, probability, guess in zip(rows, p, pred):
        grouped[(row['domain'], int(row['label']))].append(int(guess == row['label']))
    groups = [{'domain': domain, 'label': label, 'n': len(values),
               'correct': sum(values), 'accuracy': sum(values)/len(values)}
              for (domain, label), values in sorted(grouped.items())]
    output.mkdir(parents=True, exist_ok=False)
    with (output / 'predictions.csv').open('x', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['source_file', 'domain', 'group',
                                                   'label', 'image_sha256', 'prob_fake',
                                                   'pred', 'correct'])
        writer.writeheader()
        for row, probability, guess in zip(rows, p, pred):
            writer.writerow({key: row[key] for key in
                             ('source_file', 'domain', 'group', 'label', 'image_sha256')}
                            | {'prob_fake': float(probability), 'pred': int(guess),
                               'correct': int(guess == row['label'])})
    save_exclusive(output / 'metrics.json', summary)
    save_exclusive(output / 'per_group.json', groups)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print('Avaliação externa salva:', output, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('pilot', 'prepare', 'stage', 'evaluate'))
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--code-root', type=Path,
                        default=Path(os.environ.get('TCC_CODE_ROOT', '/content/tcc_codigo/ausente')))
    parser.add_argument('--cache-dir', type=Path, default=Path('/content/mirage_source_cache'))
    parser.add_argument('--local-dir', type=Path, default=Path('/content') / NAME)
    args = parser.parse_args()
    if args.mode == 'pilot':
        pilot(args)
    elif args.mode == 'prepare':
        prepare(args)
    elif args.mode == 'stage':
        stage(args)
    else:
        evaluate(args)


if __name__ == '__main__':
    main()
