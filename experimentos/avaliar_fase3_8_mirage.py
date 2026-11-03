"""Avaliação externa comparativa da Fase 3.8 no Mirage 2000 já congelado.

`freeze` registra pesos, calibração, código e manifesto antes da inferência.
`evaluate` usa apenas esses valores e grava resultado novo. Não treina nem
recalibra. O Mirage já foi observado na 3.5: este teste é comparativo, não
uma avaliação cega independente para desenvolvimento posterior.
"""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
import os
import re
from pathlib import Path, PurePosixPath
import tempfile


ROOT = Path('/content/drive/MyDrive/TCC_Final')
NAME = 'benchmark_externo_mirage_2000_v1'
RUN = 'fase3_8_aigi_pares256_v1'
PREVIOUS = 'fase3_7_cfsmall_expansao_v1'
TOTAL = 2000
MANIFEST_SHA256 = '4b3658f880e022957557933fd662c69485c19a5ed9a04e2e09c32ce485737190'
PROTOCOL = 'five_crop_224; upscale_bicubic_only_if_short_side_below_224; mean_logits'


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def paths(root):
    benchmark = root / 'dados_externos' / NAME / 'manifest.json'
    run = root / 'modelos' / RUN
    lock = root / 'resultados' / NAME / 'locks' / f'{RUN}.json'
    result = root / 'resultados' / NAME / RUN
    previous = root / 'resultados' / NAME / PREVIOUS
    return benchmark, run, lock, result, previous


def checked_rows(manifest):
    if not manifest.is_file() or digest(manifest) != MANIFEST_SHA256:
        raise ValueError('Manifesto Mirage difere do conjunto já avaliado pela 3.5')
    data = json.loads(manifest.read_text(encoding='utf-8'))
    rows = data.get('rows', [])
    if (len(rows) != TOTAL or Counter(row['label'] for row in rows) != {0: 1000, 1: 1000}
            or not data.get('exact_overlap_checked')):
        raise ValueError('Contagens ou auditoria de sobreposição Mirage inválidas')
    names = set()
    for row in rows:
        value = row.get('image')
        if not isinstance(value, str) or '\\' in value:
            raise ValueError('Caminho de imagem inválido no manifesto')
        rel = PurePosixPath(value)
        if rel.is_absolute() or '..' in rel.parts or not rel.parts or rel.parts[0] != 'images':
            raise ValueError(f'Caminho inseguro: {value}')
        if value in names:
            raise ValueError(f'Imagem repetida no manifesto: {value}')
        names.add(value)
        if not isinstance(row.get('image_sha256'), str) or not re.fullmatch(r'[0-9a-f]{64}', row['image_sha256']):
            raise ValueError('Hash de imagem ausente')
    return rows


def freeze_payload(root, code_root):
    """Confere os artefatos internos concluídos, sem ler nenhuma predição."""
    import torch

    manifest, run, _, result, previous = paths(root)
    checked_rows(manifest)
    if result.exists():
        raise FileExistsError(f'Avaliação 3.8 já existe: {result}')
    checkpoint = run / 'phase3_final.pt'
    config_path = run / 'experiment_config.json'
    metrics_path = run / 'internal_metrics.json'
    complete_path = run / 'training_complete.json'
    architecture = code_root / 'Notebooks/fase_3/fase3.3_compressao.ipynb'
    script = Path(__file__)
    prior_metrics = previous / 'metrics.json'
    prior_predictions = previous / 'predictions.csv'
    for required in (checkpoint, config_path, metrics_path, complete_path,
                     architecture, prior_metrics, prior_predictions):
        if not required.is_file():
            raise FileNotFoundError(required)
    config = json.loads(config_path.read_text(encoding='utf-8'))
    metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
    complete = json.loads(complete_path.read_text(encoding='utf-8'))
    prior = json.loads(prior_metrics.read_text(encoding='utf-8'))
    if (config.get('RUN_NAME') != RUN or config.get('SOURCE_SHA256') != digest(architecture)
            or complete.get('status') != 'internal_training_complete'
            or metrics.get('external_used') is not False
            or prior.get('model') != PREVIOUS
            or prior.get('manifest_sha256') != MANIFEST_SHA256
            or prior.get('n') != TOTAL):
        raise ValueError('Execução interna ou resultado anterior incompatível')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    temperature, threshold = float(saved['temperature']), float(saved['threshold'])
    if (saved.get('cfg_phase3', {}).get('RUN_NAME') != RUN
            or not math.isfinite(temperature) or temperature <= 0
            or not math.isfinite(threshold) or not 0 < threshold < 1
            or abs(temperature - float(metrics['temperature'])) > 1e-8
            or abs(threshold - float(metrics['threshold'])) > 1e-8):
        raise ValueError('Temperatura, limiar ou identidade da 3.8 divergem do treino')
    return {'model': RUN, 'checkpoint_sha256': digest(checkpoint),
            'config_sha256': digest(config_path), 'internal_metrics_sha256': digest(metrics_path),
            'architecture_source_sha256': digest(architecture),
            'evaluation_script_sha256': digest(script),
            'manifest_sha256': MANIFEST_SHA256, 'prior_metrics_sha256': digest(prior_metrics),
            'prior_predictions_sha256': digest(prior_predictions),
            'temperature': temperature, 'threshold': threshold, 'protocol': PROTOCOL,
            'external_benchmark_previously_seen': True,
            'blind_independent_final_test': False}


def freeze(args):
    _, _, lock, _, _ = paths(args.root)
    payload = freeze_payload(args.root, args.code_root)
    lock.parent.mkdir(parents=True, exist_ok=True)
    if lock.is_file():
        if json.loads(lock.read_text(encoding='utf-8')) != payload:
            raise ValueError('Bloqueio anterior difere; não substitua a configuração')
        print('Bloqueio existente preservado:', lock, flush=True)
    else:
        with lock.open('x', encoding='utf-8') as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False)
        print('Configuração 3.8 congelada antes da inferência:', lock, flush=True)
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)


def wilson(correct, total):
    z = 1.959963984540054
    p = correct / total
    denominator = 1 + z*z/total
    middle = (p + z*z/(2*total)) / denominator
    half = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / denominator
    return [middle-half, middle+half]


def paired_comparison(current_rows, previous_csv):
    """Conta trocas de acerto nas mesmas imagens; não escolhe parâmetros."""
    with previous_csv.open(newline='', encoding='utf-8') as stream:
        old_rows = list(csv.DictReader(stream))
    old = {row['source_file']: int(row['correct']) for row in old_rows}
    new = {row['source_file']: int(row['correct']) for row in current_rows}
    if (len(old_rows) != TOTAL or len(current_rows) != TOTAL or
            len(old) != TOTAL or len(new) != TOTAL or old.keys() != new.keys()):
        raise ValueError('Predições anteriores não correspondem aos mesmos 2.000 arquivos')
    return {'both_correct': sum(old[key] and new[key] for key in old),
            'only_phase3_7_correct': sum(old[key] and not new[key] for key in old),
            'only_phase3_8_correct': sum(new[key] and not old[key] for key in old),
            'both_wrong': sum(not old[key] and not new[key] for key in old)}


def evaluate(args):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image, ImageOps
    from sklearn.metrics import roc_auc_score
    from torchvision import transforms
    from avaliar_benchmark_2000 import load_architecture, load_model

    source_manifest, run, lock, output, previous = paths(args.root)
    if output.exists():
        raise FileExistsError(f'Resultado 3.8 já preservado: {output}')
    if not lock.is_file():
        raise FileNotFoundError('Execute freeze antes de avaliar; bloqueio ausente')
    frozen = json.loads(lock.read_text(encoding='utf-8'))
    if frozen != freeze_payload(args.root, args.code_root):
        raise ValueError('Pesos, calibração, código ou manifesto mudaram após freeze')
    local_manifest = args.local_dir / 'manifest.json'
    if not local_manifest.is_file() or digest(local_manifest) != digest(source_manifest):
        raise ValueError('Execute stage: manifesto local difere do Drive')
    rows = checked_rows(local_manifest)
    for index, row in enumerate(rows, 1):
        path = args.local_dir / row['image']
        if not path.is_file() or digest(path) != row['image_sha256']:
            raise ValueError(f'Imagem ausente ou alterada: {row["image"]}')
        if index % 100 == 0:
            print(f'Integridade antes da inferência: {index}/{TOTAL}', flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError('Conecte GPU ao Colab para avaliar')
    device = torch.device('cuda')
    phase1_cls, phase2_cls, arch_sha = load_architecture(args.code_root, device)
    model, temperature, threshold, _ = load_model(
        RUN, run / 'phase3_final.pt', phase1_cls, phase2_cls, device)
    if temperature != frozen['temperature'] or threshold != frozen['threshold']:
        raise ValueError('Calibração do checkpoint diverge do bloqueio')
    print(f'Fase 3.8 congelada | GPU={torch.cuda.get_device_name(0)} | '
          f'T={temperature:.6f} | limiar={threshold:.6f}', flush=True)
    norm = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize((0.5,)*3, (0.5,)*3)])
    crop = transforms.FiveCrop(224)
    probabilities, upscaled = [], 0
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
                raise ValueError('Logits não finitos ou dimensão inesperada')
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
    predictions = [{**{key: row[key] for key in
                        ('source_file', 'domain', 'group', 'label', 'image_sha256')},
                    'prob_fake': float(probability), 'pred': int(guess),
                    'correct': int(guess == row['label'])}
                   for row, probability, guess in zip(rows, p, pred)]
    comparison = paired_comparison(predictions, previous / 'predictions.csv')
    summary = {'model': RUN, 'n': TOTAL, 'real': 1000, 'fake': 1000,
               'accuracy': (tn+tp)/TOTAL,
               'accuracy_ci95_wilson': wilson(tn+tp, TOTAL),
               'balanced_accuracy': (recall+specificity)/2,
               'auc': float(roc_auc_score(y, p)), 'precision': precision,
               'recall_fake': recall, 'specificity_real': specificity,
               'f1': 2*precision*recall/(precision+recall) if precision+recall else 0.0,
               'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
               'temperature_frozen': temperature, 'threshold_frozen': threshold,
               'upscaled_short_side_below_224': upscaled,
               'checkpoint_sha256': frozen['checkpoint_sha256'],
               'manifest_sha256': MANIFEST_SHA256,
               'evaluation_script_sha256': frozen['evaluation_script_sha256'],
               'architecture_source_sha256': frozen['architecture_source_sha256'],
               'architecture_extraction_sha256': arch_sha,
               'protocol': PROTOCOL, 'paired_with': PREVIOUS,
               'paired_correctness': comparison,
               'external_benchmark_previously_seen': True,
               'blind_independent_final_test': False}
    grouped = defaultdict(list)
    for row, guess in zip(rows, pred):
        grouped[(row['domain'], int(row['label']))].append(int(guess == row['label']))
    groups = [{'domain': domain, 'label': label, 'n': len(values),
               'correct': sum(values), 'accuracy': sum(values)/len(values)}
              for (domain, label), values in sorted(grouped.items())]
    output.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(tempfile.mkdtemp(prefix=f'{RUN}.incompleto.', dir=output.parent))
    with (staged / 'predictions.csv').open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    for name, payload in (('metrics.json', summary), ('per_group.json', groups)):
        with (staged / name).open('x', encoding='utf-8') as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False)
    if output.exists():
        raise FileExistsError(f'Resultado 3.8 já preservado: {output}')
    staged.rename(output)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print('Avaliação externa comparativa salva:', output, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('freeze', 'evaluate'))
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--code-root', type=Path, default=Path(
        os.environ.get('TCC_CODE_ROOT', '/content/tcc_codigo/ausente')))
    parser.add_argument('--local-dir', type=Path, default=Path('/content') / NAME)
    args = parser.parse_args()
    if args.mode == 'freeze':
        freeze(args)
    else:
        evaluate(args)


if __name__ == '__main__':
    main()
