"""Audita a Fase 3.9 na seleção interna, sem treino ou avaliação externa.

Compara checkpoints 3.8 e 3.9 nas mesmas imagens de seleção. Lê cabeçalhos
somente de treino/seleção; calibração e OpenFake reservado são conferidos pelo
manifesto de divisões, sem abrir suas imagens. O Mirage não é acessado.
"""

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path

from amostragem_fase3_4 import balanced_sqrt_group_weights
from auditoria_interna_fase3_8 import (
    collect_metadata, key, metadata_index, segment_table, sha256, source_rows,
    verify_partitions, write_csv,
)
from diagnostico_interno_fase3_5 import summarize


ROOT = Path('/content/drive/MyDrive/TCC_Final')
CODE_ROOT = Path(os.environ.get('TCC_CODE_ROOT', '/content/tcc_codigo/ausente'))
RUN38 = 'fase3_8_aigi_pares256_v1'
RUN39 = 'fase3_9_tgif_sd2fr_v1'
CHECKPOINT_SHA256 = {
    RUN38: 'de4a11cbb733ad9088860d5f254cbb2a900f181cc1ccb28b2fd894cf4a37706a',
    RUN39: '529ef195be38b57ac2f876e7733141e26e7593644be9d83d0da56205319fbb01',
}
OUTPUT_NAME = 'auditoria_interna_fase3_9_v1'
SOURCES = (
    ('PRIOR', 'dados_fase3_5/fase3_5_novos_v1/manifest.json',
     Path('/content/fase3_5_novos_v1'), 'cfsmall'),
    ('NEW', 'dados_fase3_7/fase3_7_cfsmall_expansao_v1/manifest.json',
     Path('/content/fase3_7_cfsmall_expansao_v1'), 'cfsmall'),
    ('AIGI', 'dados_fase3_8/aigi_pares_256_v1/manifest.json',
     Path('/content/aigi_pares_256_v1'), 'aigi'),
    ('TGIF', 'dados_fase3_9/tgif_sd2fr_768_v1/manifest.json',
     Path('/content/tgif_sd2fr_768_v1'), 'tgif'),
)
PARTITIONS = ('train', 'validation_selection', 'validation_calibration',
              'test_reserved')


def compare_splits(split38, split39):
    """A única inclusão permitida entre 3.8 e 3.9 é o bloco TGIF."""
    counts = {}
    for subset in PARTITIONS:
        before = Counter(key(row) for row in split38[subset])
        after = Counter(key(row) for row in split39[subset])
        if any(count != 1 for count in after.values()):
            raise ValueError(f'Caminho repetido na 3.9: {subset}')
        added = after - before
        if subset == 'test_reserved':
            if before != after:
                raise ValueError('Teste OpenFake reservado mudou entre 3.8 e 3.9')
        elif before - after or any(not group.startswith('tgif:')
                                   for _, _, group in added):
            raise ValueError(f'Partição histórica mudou além dos pares TGIF: {subset}')
        counts[subset] = {'phase3_8': sum(before.values()),
                          'phase3_9': sum(after.values()),
                          'tgif_added': sum(added.values())}
    expected = {'train': 1024, 'validation_selection': 256,
                'validation_calibration': 256, 'test_reserved': 0}
    if any(counts[name]['tgif_added'] != amount for name, amount in expected.items()):
        raise ValueError('Contagens TGIF diferentes do plano congelado')
    return counts


def verify_tgif_pairs(manifest):
    """O real e o sintético de cada COCO ID devem ficar na mesma divisão."""
    pairs = {}
    for row in manifest['rows']:
        pairs.setdefault(str(row['image_id']), []).append(row)
    if len(pairs) != 768 or len(manifest['rows']) != 1536:
        raise ValueError('Manifesto TGIF com quantidade inesperada')
    for image_id, rows in pairs.items():
        if (len(rows) != 2 or {int(row['label']) for row in rows} != {0, 1}
                or len({row['subset'] for row in rows}) != 1
                or len({row['category'] for row in rows}) != 1):
            raise ValueError(f'Par TGIF inválido ou dividido: {image_id}')
    return Counter(rows[0]['subset'] for rows in pairs.values())


def build_index(old, sources):
    """Anexa categoria TGIF sem confundir suas fotos com Community Forensics."""
    index = metadata_index(old, sources[:-1])
    for item in index.values():
        item.update(pair_id='', category='')
    manifest, local, prefix = sources[-1]
    for _, item, row in source_rows(manifest, local, prefix):
        identity = key(item)
        if identity in index:
            raise ValueError('Imagem TGIF repetida nos manifestos internos')
        size = row.get('original_size')
        index[identity] = {
            'dataset': 'TGIF', 'original_format': row.get('original_format') or 'não_registrado',
            'real_source': 'COCO' if int(row['label']) == 0 else '',
            'original_short_side': min(size) if size else '',
            'pair_id': str(row['image_id']), 'category': str(row['category']),
        }
    return index


def load_architecture(source):
    """Reusa a extração de arquitetura e transforms já validada na 3.8."""
    notebook = json.loads(source.read_text(encoding='utf-8'))
    scope = {'__name__': '__audit39__'}
    for number, prefix in ((3, None), (45, None), (47, 'train_pool ='),
                           (49, '# ---------- carrega o checkpoint final da Fase 2 ----------')):
        code = ''.join(notebook['cells'][number]['source'])
        if prefix:
            if prefix not in code:
                raise RuntimeError(f'Estrutura histórica da célula {number} mudou')
            code = code.split(prefix, 1)[0]
        exec(compile(code, f'{source.name}:cell{number}', 'exec'), scope)
    return scope


def infer_selection(rows, scope, checkpoints, configs):
    """Cinco recortes e T/limiar próprios de cada checkpoint, sem ajuste."""
    import torch
    from torch.utils.data import DataLoader

    if not torch.cuda.is_available():
        raise RuntimeError('Conecte GPU ao Colab para inferir a seleção interna')
    loader = DataLoader(scope['PathDataset'](rows, scope['multicrop_tf']),
                        batch_size=8, shuffle=False,
                        num_workers=min(2, os.cpu_count() or 1), pin_memory=True)
    results = {}
    for name in (RUN38, RUN39):
        saved = torch.load(checkpoints[name], map_location='cpu', weights_only=False)
        if (saved.get('cfg_phase3', {}).get('RUN_NAME') != name
                or saved['cfg_phase3'].get('SOURCE_SHA256') != configs[name]['SOURCE_SHA256']):
            raise ValueError(f'Checkpoint não corresponde ao treino {name}')
        temperature = float(saved['temperature'])
        threshold = float(saved['threshold'])
        if not (math.isfinite(temperature) and temperature > 0
                and math.isfinite(threshold) and 0 < threshold < 1):
            raise ValueError(f'Calibração inválida: {name}')
        model = scope['Phase2Detector'](
            branches=saved['cfg']['BRANCHES'], fusion=saved['cfg']['FUSION'],
            proj_dim=saved['cfg']['PROJ_DIM'], pretrained_cnn=False)
        model.load_state_dict(saved['model_state_dict'], strict=True)
        model = model.cuda().eval()
        probabilities = []
        with torch.inference_mode():
            for images, _ in loader:
                count = images.shape[0]
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    logits = model(images.cuda(non_blocking=True).flatten(0, 1))
                mean_logits = logits.float().view(count, 5, 2).mean(1)
                if not torch.isfinite(mean_logits).all():
                    raise ValueError(f'Logits não finitos em {name}')
                probabilities.extend(torch.softmax(mean_logits / temperature, 1)[:, 1]
                                     .cpu().tolist())
                if len(probabilities) // 100 != (len(probabilities)-count) // 100 \
                        or len(probabilities) == len(rows):
                    print(f'{name}: {len(probabilities)}/{len(rows)} inferências internas',
                          flush=True)
        del model, saved
        torch.cuda.empty_cache()
        if len(probabilities) != len(rows):
            raise ValueError(f'Predições incompletas: {name}')
        results[name] = {'probabilities': probabilities, 'temperature': temperature,
                         'threshold': threshold}
    return results


def subset_metrics(rows, probabilities, threshold):
    from sklearn.metrics import roc_auc_score

    metrics = summarize(rows, probabilities, threshold)
    metrics['auc'] = float(roc_auc_score([int(row['label']) for row in rows], probabilities))
    return metrics


def compare_predictions(rows, predicted38, predicted39, indices):
    counts = Counter()
    for index in indices:
        correct38 = int(predicted38[index] == int(rows[index]['label']))
        correct39 = int(predicted39[index] == int(rows[index]['label']))
        counts[(correct38, correct39)] += 1
    return {'both_correct': counts[1, 1], 'only_phase3_8_correct': counts[1, 0],
            'only_phase3_9_correct': counts[0, 1], 'both_wrong': counts[0, 0]}


def sampler_mass(rows, index):
    """Calcula quanto o sampler de treino espera sortear de cada fonte/classe."""
    weights = balanced_sqrt_group_weights(rows)
    total = sum(weights)
    if not math.isclose(total, 1.0, rel_tol=1e-6):
        raise ValueError('Pesos de amostragem não somam um')
    counts = Counter()
    mass = Counter()
    for row, weight in zip(rows, weights):
        dataset = index[key(row)]['dataset']
        label = int(row['label'])
        item = dataset, label, str(row['group'])
        counts[item] += 1
        mass[item] += weight
    return [{'dataset': dataset, 'label': label, 'group': group,
             'image_count': counts[dataset, label, group],
             'sampling_mass': value,
             'expected_draws_per_epoch': value * len(rows)}
            for (dataset, label, group), value in sorted(mass.items())]


def audit(args):
    root = args.root
    output = root / 'resultados' / OUTPUT_NAME
    if output.exists():
        summary = output / 'summary.json'
        if summary.is_file():
            print('Auditoria já preservada:', summary, flush=True)
            print(summary.read_text(encoding='utf-8'), flush=True)
            return
        raise FileExistsError(f'Diretório parcial; verificar sem sobrescrever: {output}')

    source = args.code_root / 'Notebooks/fase_3/fase3.3_compressao.ipynb'
    scripts = {RUN38: args.code_root / 'experimentos/fase3_8_aigi_finetune.py',
               RUN39: args.code_root / 'experimentos/fase3_9_tgif_finetune.py'}
    old_path = root / 'dados_fase3_4/manifest.json'
    runs = {name: root / 'modelos' / name for name in (RUN38, RUN39)}
    checkpoints = {name: path / 'phase3_final.pt' for name, path in runs.items()}
    paths = [source, old_path, *scripts.values(), *checkpoints.values()]
    for run in runs.values():
        paths.extend(run / filename for filename in
                     ('experiment_config.json', 'split_manifest.json',
                      'internal_metrics.json', 'training_complete.json'))
    paths.extend(local / 'manifest.json' for _, _, local, _ in SOURCES)
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    configs = {name: json.loads((run / 'experiment_config.json').read_text(encoding='utf-8'))
               for name, run in runs.items()}
    internals = {name: json.loads((run / 'internal_metrics.json').read_text(encoding='utf-8'))
                 for name, run in runs.items()}
    splits = {name: json.loads((run / 'split_manifest.json').read_text(encoding='utf-8'))
              for name, run in runs.items()}
    for name, run in runs.items():
        complete = json.loads((run / 'training_complete.json').read_text(encoding='utf-8'))
        cfg = configs[name]
        if complete.get('status') != 'internal_training_complete' or cfg.get('RUN_NAME') != name:
            raise ValueError(f'Treino interno incompleto ou versão incorreta: {name}')
        if (sha256(checkpoints[name]) != CHECKPOINT_SHA256[name]
                or sha256(source) != cfg['SOURCE_SHA256']
                or sha256(scripts[name]) != cfg['SCRIPT_SHA256']
                or sha256(old_path) != cfg['DATA_MANIFEST_SHA256']):
            raise ValueError(f'Checkpoint, fonte ou manifesto diferente do treino: {name}')
        for field, sha_field in (('manifest_sha256', 'DATA_MANIFEST_SHA256'),
                                 ('prior_manifest_sha256', 'PRIOR_DATA_MANIFEST_SHA256'),
                                 ('new_manifest_sha256', 'NEW_DATA_MANIFEST_SHA256'),
                                 ('aigi_manifest_sha256', 'AIGI_DATA_MANIFEST_SHA256')):
            if splits[name].get(field) != cfg[sha_field]:
                raise ValueError(f'Divisão difere da configuração: {name}, {field}')
    if splits[RUN39].get('tgif_manifest_sha256') != configs[RUN39]['TGIF_DATA_MANIFEST_SHA256']:
        raise ValueError('Hash TGIF ausente ou incorreto no split 3.9')

    old = json.loads(old_path.read_text(encoding='utf-8'))
    sources = []
    for label, relative, local, prefix in SOURCES:
        path = local / 'manifest.json'
        digest = sha256(path)
        if digest != configs[RUN39][label + '_DATA_MANIFEST_SHA256']:
            raise ValueError(f'Manifesto {label} difere do treino 3.9')
        if label != 'TGIF' and digest != configs[RUN38][label + '_DATA_MANIFEST_SHA256']:
            raise ValueError(f'Manifesto {label} difere do treino 3.8')
        sources.append((json.loads(path.read_text(encoding='utf-8')), local, prefix))
    counts38 = verify_partitions(splits[RUN38], old, sources[:-1])
    counts39 = verify_partitions(splits[RUN39], old, sources)
    growth = compare_splits(splits[RUN38], splits[RUN39])
    pair_counts = verify_tgif_pairs(sources[-1][0])
    if pair_counts != {'train': 512, 'validation_selection': 128,
                       'validation_calibration': 128}:
        raise ValueError('Divisão TGIF por pares difere do plano')
    print('Partições 3.8/3.9 conferidas:', counts38, counts39, flush=True)
    print('TGIF adicional por divisão:', growth, flush=True)
    print('Calibração e teste reservado não serão inferidos.', flush=True)

    index = build_index(old, sources)
    sampling = sampler_mass(splits[RUN39]['train'], index)
    records = collect_metadata(splits[RUN39], index)
    selection = [dict(row) for row in records if row['subset'] == 'validation_selection']
    split_rows = splits[RUN39]['validation_selection']
    if [key(row) for row in selection] != [key(row) for row in split_rows]:
        raise ValueError('Metadados e seleção fora de ordem')
    scope = load_architecture(source)
    predictions = infer_selection(split_rows, scope, checkpoints, configs)
    for name in (RUN38, RUN39):
        if (abs(predictions[name]['temperature'] - float(internals[name]['temperature'])) > 1e-8
                or abs(predictions[name]['threshold'] - float(internals[name]['threshold'])) > 1e-8):
            raise ValueError(f'Calibração congelada difere das métricas internas: {name}')
    masks = {'common_3_8': [i for i, row in enumerate(selection) if row['dataset'] != 'TGIF'],
             'tgif_new': [i for i, row in enumerate(selection) if row['dataset'] == 'TGIF'],
             'all_3_9': list(range(len(selection)))}
    if (len(masks['common_3_8']) != 1420 or len(masks['tgif_new']) != 256
            or any({selection[i]['label'] for i in indices} != {0, 1}
                   for indices in masks.values())):
        raise ValueError('Seleção comum/TGIF com contagem ou classe inesperada')
    metrics = {}
    predicted = {}
    for name in (RUN38, RUN39):
        probs = predictions[name]['probabilities']
        threshold = predictions[name]['threshold']
        predicted[name] = [int(prob >= threshold) for prob in probs]
        metrics[name] = {part: subset_metrics([split_rows[i] for i in indices],
                                             [probs[i] for i in indices], threshold)
                         for part, indices in masks.items()}
        print(f'{name}: seleção comum AUC={metrics[name]["common_3_8"]["auc"]:.4f} '
              f'| TGIF AUC={metrics[name]["tgif_new"]["auc"]:.4f}', flush=True)
    paired = {part: compare_predictions(split_rows, predicted[RUN38], predicted[RUN39], indices)
              for part, indices in masks.items()}
    for i, row in enumerate(selection):
        for name, suffix in ((RUN38, '3_8'), (RUN39, '3_9')):
            row[f'probability_fake_{suffix}'] = predictions[name]['probabilities'][i]
            row[f'prediction_{suffix}'] = predicted[name][i]
            row[f'error_{suffix}'] = int(predicted[name][i] != int(row['label']))
        row['error_type'] = ('FP' if row['label'] == 0 else 'FN') \
            if row['error_3_9'] else ''
    errors = [row for row in selection if row['error_3_9']]
    distribution = segment_table(records, ('subset', 'dataset', 'group', 'category',
                                           'original_format', 'processed_size_band',
                                           'dataset_size_band', 'processed_aspect_band'))
    segments = segment_table(selection, ('dataset', 'group', 'category',
                                         'original_format', 'processed_size_band',
                                         'dataset_size_band', 'processed_aspect_band'),
                             errors=True)
    summary = {
        'run_name': RUN39, 'counts_phase3_8': counts38, 'counts_phase3_9': counts39,
        'tgif_growth': growth, 'tgif_pairs_by_partition': dict(pair_counts),
        'sampling_mass_by_dataset_class': [
            {'dataset': dataset, 'label': label, 'sampling_mass': sum(
                item['sampling_mass'] for item in sampling
                if item['dataset'] == dataset and item['label'] == label)}
            for dataset, label in sorted({(item['dataset'], item['label']) for item in sampling})],
        'selection_metrics_descriptive': metrics, 'paired_correctness_internal': paired,
        'frozen_calibration': {name: {'temperature': predictions[name]['temperature'],
                                     'threshold': predictions[name]['threshold']}
                               for name in (RUN38, RUN39)},
        'checkpoint_sha256': {name: sha256(checkpoints[name]) for name in (RUN38, RUN39)},
        'split_manifest_sha256': {name: sha256(runs[name] / 'split_manifest.json')
                                  for name in (RUN38, RUN39)},
        'architecture_source_sha256': sha256(source),
        'external_used': False, 'calibration_inferred': False,
        'openfake_test_inferred': False,
        'note': 'Ambos os checkpoints foram selecionados em subconjuntos destas imagens; métricas descritivas, sem inferência causal ou prova externa. Mirage não foi lido.'
    }
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / 'metadata_train_selection.csv', records)
    write_csv(output / 'distribution.csv', distribution)
    write_csv(output / 'sampler_mass_train.csv', sampling)
    write_csv(output / 'predictions_selection_paired.csv', selection)
    write_csv(output / 'errors_selection_3_9.csv', errors, fieldnames=selection[0].keys())
    write_csv(output / 'segments_selection_3_9.csv', segments)
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
