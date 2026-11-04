"""Auditoria imutável da Fase 3.8: dados de treino e erros na seleção interna.

Não treina, não recalibra e não lê imagens ou predições externas. A calibração
e o teste OpenFake reservado são conferidos apenas pelo registro das partições.
"""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path

from diagnostico_interno_fase3_5 import summarize


ROOT = Path('/content/drive/MyDrive/TCC_Final')
CODE_ROOT = Path(os.environ.get('TCC_CODE_ROOT', '/content/tcc_codigo/ausente'))
RUN_NAME = 'fase3_8_aigi_pares256_v1'
OUTPUT_NAME = 'auditoria_interna_fase3_8_v1'
EXPECTED_CHECKPOINT_SHA256 = 'de4a11cbb733ad9088860d5f254cbb2a900f181cc1ccb28b2fd894cf4a37706a'
NEW_SOURCES = (
    ('prior', 'dados_fase3_5/fase3_5_novos_v1/manifest.json', Path('/content/fase3_5_novos_v1'), 'cfsmall'),
    ('new', 'dados_fase3_7/fase3_7_cfsmall_expansao_v1/manifest.json', Path('/content/fase3_7_cfsmall_expansao_v1'), 'cfsmall'),
    ('aigi', 'dados_fase3_8/aigi_pares_256_v1/manifest.json', Path('/content/aigi_pares_256_v1'), 'aigi'),
)


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def key(row):
    return str(row['path']), int(row['label']), str(row['group'])


def source_rows(manifest, local, prefix):
    """Traduz um manifesto congelado para os caminhos usados pelo treinador."""
    for row in manifest['rows']:
        relative = Path(row['image'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Caminho inseguro no manifesto interno')
        subset = row['subset']
        if subset not in ('train', 'validation_selection', 'validation_calibration'):
            raise ValueError('Divisão interna desconhecida')
        yield subset, {'path': str(local / relative), 'label': int(row['label']),
                       'group': f"{prefix}:{row['group']}"}, row


def verify_partitions(split, old, sources):
    """Confere integralmente as divisões, sem abrir calibração ou teste."""
    expected = Counter(key(row) for row in old['train'])
    by_subset = {name: Counter(key(row) for row in split[name])
                 for name in ('train', 'validation_selection',
                              'validation_calibration')}
    for manifest, local, prefix in sources:
        for subset, item, _ in source_rows(manifest, local, prefix):
            k = key(item)
            expected[k] += 1
            if by_subset[subset][k] != 1:
                raise ValueError('Imagem nova atravessa divisões internas')
    actual = sum(by_subset.values(), Counter())
    if actual != expected or any(count != 1 for count in actual.values()):
        raise ValueError('Partições internas diferem dos manifestos congelados')
    if Counter(key(row) for row in split['test_reserved']) != Counter(
            key(row) for row in old['test']):
        raise ValueError('Registro do teste reservado mudou')
    return {subset: len(split[subset]) for subset in
            ('train', 'validation_selection', 'validation_calibration', 'test_reserved')}


def metadata_index(old, sources):
    index = {}
    for row in old['train']:
        index[key(row)] = {'dataset': 'OpenFake', 'original_format':
                           str(row.get('orig_format') or 'não_registrado').upper(),
                           'real_source': str(row['group']) if int(row['label']) == 0 else '',
                           'original_short_side': min(row['orig_w'], row['orig_h'])
                           if row.get('orig_w') and row.get('orig_h') else ''}
    for manifest, local, prefix in sources:
        for _, item, row in source_rows(manifest, local, prefix):
            k = key(item)
            if k in index:
                raise ValueError('Imagem repetida nos manifestos internos')
            size = row.get('original_size') or row.get('real_original_size')
            index[k] = {'dataset': 'AIGI' if prefix == 'aigi' else 'CommunityForensics-Small',
                        'original_format': str(row.get('original_format') or
                                               row.get('real_original_format') or
                                               'não_registrado').upper(),
                        'real_source': str(row.get('real_source') or 'não_registrado')
                        if int(row['label']) == 0 else '',
                        'original_short_side': min(size) if size else ''}
    return index


def size_band(short):
    return '<256' if short < 256 else ('256-511' if short < 512 else '512+')


def aspect_band(width, height):
    ratio = width / height
    return 'vertical' if ratio < 0.8 else ('horizontal' if ratio > 1.25 else 'quase_quadrada')


def collect_metadata(split, index):
    from PIL import Image

    records = []
    for subset in ('train', 'validation_selection'):
        rows = split[subset]
        for number, row in enumerate(rows, 1):
            k = key(row)
            if k not in index:
                raise ValueError('Linha interna sem origem no manifesto')
            path = Path(row['path'])
            with Image.open(path) as image:
                width, height = image.size
                processed_format = image.format
            if processed_format != 'JPEG':
                raise ValueError(f'Imagem interna processada não é JPEG: {path}')
            short = min(width, height)
            size_group = size_band(short)
            records.append({'subset': subset, 'path': str(path),
                            'label': int(row['label']), 'group': str(row['group']),
                            **index[k], 'processed_format': processed_format,
                            'processed_width': width, 'processed_height': height,
                            'processed_short_side': short,
                            'processed_size_band': size_group,
                            'dataset_size_band': f"{index[k]['dataset']} | {size_group}",
                            'subset_dataset_size_band':
                            f"{subset} | {index[k]['dataset']} | {size_group}",
                            'processed_aspect_band': aspect_band(width, height),
                            'probability_fake': '', 'prediction': '',
                            'error_type': ''})
            if number % 500 == 0 or number == len(rows):
                print(f'Metadados {subset}: {number}/{len(rows)}', flush=True)
    return records


def infer_selection(split, source, config, checkpoint):
    """Aplica cinco recortes e parâmetros congelados só na seleção interna."""
    import torch
    from torch.utils.data import DataLoader
    from sklearn.metrics import roc_auc_score

    if not torch.cuda.is_available():
        raise RuntimeError('Conecte GPU ao Colab para inferir a seleção interna')
    notebook = json.loads(source.read_text(encoding='utf-8'))
    scope = {'__name__': '__audit__'}
    for number, prefix in ((3, None), (45, None), (47, 'train_pool ='),
                           (49, '# ---------- carrega o checkpoint final da Fase 2 ----------')):
        code = ''.join(notebook['cells'][number]['source'])
        if prefix:
            if prefix not in code:
                raise RuntimeError(f'Estrutura histórica da célula {number} mudou')
            code = code.split(prefix, 1)[0]
        exec(compile(code, f'{source.name}:cell{number}', 'exec'), scope)
        if number == 3:
            scope['CFG'] = config
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if (saved.get('cfg_phase3', {}).get('RUN_NAME') != RUN_NAME or
            saved['cfg_phase3'].get('SOURCE_SHA256') != config['SOURCE_SHA256']):
        raise ValueError('Checkpoint não pertence ao treino 3.8 registrado')
    temperature, threshold = float(saved['temperature']), float(saved['threshold'])
    if not (math.isfinite(temperature) and temperature > 0 and 0 < threshold < 1):
        raise ValueError('Calibração congelada inválida')
    model = scope['Phase2Detector'](branches=saved['cfg']['BRANCHES'],
                                    fusion=saved['cfg']['FUSION'],
                                    proj_dim=saved['cfg']['PROJ_DIM'],
                                    pretrained_cnn=False)
    model.load_state_dict(saved['model_state_dict'], strict=True)
    model = model.cuda().eval()
    rows = split['validation_selection']
    loader = DataLoader(scope['PathDataset'](rows, scope['multicrop_tf']),
                        batch_size=8, shuffle=False,
                        num_workers=min(2, os.cpu_count() or 1), pin_memory=True)
    probabilities = []
    processed = 0
    with torch.inference_mode():
        for images, _ in loader:
            count = images.shape[0]
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                logits = model(images.cuda(non_blocking=True).flatten(0, 1))
            scores = logits.float().view(count, 5, 2).mean(1)
            if not torch.isfinite(scores).all():
                raise ValueError('Logits não finitos na seleção')
            probabilities.extend(torch.softmax(scores / temperature, 1)[:, 1].cpu().tolist())
            processed += count
            if processed // 100 != (processed-count) // 100 or processed == len(rows):
                print(f'Seleção interna: {processed}/{len(rows)} inferências', flush=True)
    if len(probabilities) != len(rows):
        raise ValueError('Predições internas incompletas')
    metrics = summarize(rows, probabilities, threshold)
    metrics['auc'] = float(roc_auc_score([int(row['label']) for row in rows], probabilities))
    return probabilities, temperature, threshold, metrics


def segment_table(records, fields, *, errors=False):
    groups = defaultdict(list)
    for row in records:
        for field in fields:
            groups[(field, str(row[field]), int(row['label']))].append(row)
    table = []
    for (field, value, label), members in sorted(groups.items()):
        wrong = sum(bool(row.get('error_type')) for row in members) if errors else 0
        table.append({'dimension': field, 'value': value, 'label': label,
                      'n': len(members), 'errors': wrong,
                      'error_rate': wrong / len(members) if errors else '',
                      'false_positives': wrong if errors and label == 0 else 0,
                      'false_negatives': wrong if errors and label == 1 else 0})
    return table


def write_csv(path, rows, *, fieldnames=None):
    if not rows and fieldnames is None:
        raise ValueError('Não há colunas para registrar')
    with path.open('x', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def audit(args):
    root = args.root
    run = root / 'modelos' / RUN_NAME
    output = root / 'resultados' / OUTPUT_NAME
    if output.exists():
        summary = output / 'summary.json'
        if summary.is_file():
            print('Auditoria já preservada:', summary, flush=True)
            print(summary.read_text(encoding='utf-8'), flush=True)
            return
        raise FileExistsError(f'Diretório parcial; verificar sem sobrescrever: {output}')
    source = args.code_root / 'Notebooks/fase_3/fase3.3_compressao.ipynb'
    script = args.code_root / 'experimentos/fase3_8_aigi_finetune.py'
    old_path = root / 'dados_fase3_4/manifest.json'
    split_path = run / 'split_manifest.json'
    checkpoint = run / 'phase3_final.pt'
    required = [source, script, old_path, split_path, checkpoint,
                run / 'experiment_config.json', run / 'internal_metrics.json',
                run / 'training_complete.json']
    required += [local / 'manifest.json' for _, _, local, _ in NEW_SOURCES]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    config = json.loads((run / 'experiment_config.json').read_text(encoding='utf-8'))
    internal = json.loads((run / 'internal_metrics.json').read_text(encoding='utf-8'))
    complete = json.loads((run / 'training_complete.json').read_text(encoding='utf-8'))
    if complete.get('status') != 'internal_training_complete' or config.get('RUN_NAME') != RUN_NAME:
        raise ValueError('Treino interno 3.8 incompleto ou de outra versão')
    if sha256(checkpoint) != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError('Checkpoint 3.8 difere do hash preservado na avaliação')
    if sha256(source) != config['SOURCE_SHA256'] or sha256(script) != config['SCRIPT_SHA256']:
        raise ValueError('Fontes do treino diferem dos hashes registrados')
    if sha256(old_path) != config['DATA_MANIFEST_SHA256']:
        raise ValueError('Manifesto OpenFake difere do treino')
    old = json.loads(old_path.read_text(encoding='utf-8'))
    sources = []
    for name, _, local, prefix in NEW_SOURCES:
        path = local / 'manifest.json'
        if sha256(path) != config[name.upper() + '_DATA_MANIFEST_SHA256']:
            raise ValueError(f'Manifesto {name} difere do treino')
        sources.append((json.loads(path.read_text(encoding='utf-8')), local, prefix))
    split = json.loads(split_path.read_text(encoding='utf-8'))
    for name, field in (('manifest_sha256', 'DATA_MANIFEST_SHA256'),
                        ('prior_manifest_sha256', 'PRIOR_DATA_MANIFEST_SHA256'),
                        ('new_manifest_sha256', 'NEW_DATA_MANIFEST_SHA256'),
                        ('aigi_manifest_sha256', 'AIGI_DATA_MANIFEST_SHA256')):
        if split.get(name) != config[field]:
            raise ValueError('Registro de partições difere do treino')
    counts = verify_partitions(split, old, sources)
    print('Partições conferidas:', counts, '| calibração e teste sem inferência', flush=True)
    index = metadata_index(old, sources)
    records = collect_metadata(split, index)
    probabilities, temperature, threshold, metrics = infer_selection(
        split, source, config, checkpoint)
    if (abs(temperature - float(internal['temperature'])) > 1e-8 or
            abs(threshold - float(internal['threshold'])) > 1e-8):
        raise ValueError('Calibração do checkpoint difere das métricas internas')
    selection = [row for row in records if row['subset'] == 'validation_selection']
    if len(selection) != len(probabilities):
        raise ValueError('Metadados e predições fora de ordem')
    for row, prob in zip(selection, probabilities):
        row['probability_fake'] = prob
        row['prediction'] = int(prob >= threshold)
        row['error_type'] = ('FP' if row['label'] == 0 else 'FN') if row['prediction'] != row['label'] else ''
    errors = [row for row in selection if row['error_type']]
    distribution = segment_table(records, ('subset', 'dataset', 'group',
                                          'original_format', 'processed_size_band',
                                          'dataset_size_band', 'subset_dataset_size_band',
                                          'processed_aspect_band'))
    segments = segment_table(selection, ('dataset', 'group', 'real_source',
                                         'original_format', 'processed_size_band',
                                         'dataset_size_band', 'processed_aspect_band'),
                             errors=True)
    print(f"Seleção: n={metrics['n']} FP={metrics['fp']} FN={metrics['fn']} "
          f"AUC={metrics['auc']:.4f}", flush=True)
    for item in sorted((row for row in segments if row['dimension'] in
                        ('dataset', 'processed_size_band')), key=lambda x: (x['dimension'], x['label'], x['value'])):
        print(f"  {item['dimension']}={item['value']} classe={item['label']} "
              f"erros={item['errors']}/{item['n']}", flush=True)
    summary = {'run_name': RUN_NAME, 'counts': counts,
               'selection_metrics_descriptive': metrics,
               'temperature_frozen': temperature, 'threshold_frozen': threshold,
               'checkpoint_sha256': sha256(checkpoint),
               'split_manifest_sha256': sha256(split_path),
               'architecture_source_sha256': sha256(source),
               'training_script_sha256': sha256(script),
               'external_used': False, 'calibration_inferred': False,
               'openfake_test_inferred': False,
               'note': 'Seleção usada para escolher o checkpoint; resultados descritivos, sem prova causal ou generalização externa. Metadados de treino/seleção; sem imagens de calibração/teste abertas.'}
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / 'metadata_train_selection.csv', records)
    write_csv(output / 'distribution.csv', distribution)
    write_csv(output / 'predictions_selection.csv', selection)
    write_csv(output / 'errors_selection.csv', errors,
              fieldnames=selection[0].keys())
    write_csv(output / 'segments_selection.csv', segments)
    with (output / 'summary.json').open('x', encoding='utf-8') as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print('Auditoria interna preservada:', output, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--code-root', type=Path, default=CODE_ROOT)
    audit(parser.parse_args())


if __name__ == '__main__':
    main()
