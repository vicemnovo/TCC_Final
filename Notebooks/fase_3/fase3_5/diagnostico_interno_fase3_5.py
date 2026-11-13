"""Diagnóstico congelado das partições internas da Fase 3.5, sem treino/teste externo."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path


ROOT = Path('/content/drive/MyDrive/TCC_Final')
CODE_ROOT = Path(os.environ.get('TCC_CODE_ROOT', '/content/tcc_codigo/ausente'))
SOURCE = CODE_ROOT / 'Notebooks/fase_3/fase3_3/fase3.3_compressao.ipynb'
LOCAL = Path('/content/fase3_5_novos_v1')
RUN = ROOT / 'modelos/fase3_5_cfsmall_v1'
OUTPUT = ROOT / 'resultados/diagnostico_interno_fase3_5_v1'
REGISTERED_PARTITIONS = ('validation_selection', 'validation_calibration')
DIAGNOSTIC_PARTITIONS = ('validation_selection',)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def row_key(row):
    return str(row['path']), int(row['label']), str(row['group'])


def expected_partitions(split, old, new, local_dir=LOCAL):
    """Confere que o registro interno contém exatamente os exemplos previstos."""
    actual = Counter(row_key(row) for subset in ('train', *REGISTERED_PARTITIONS)
                     for row in split[subset])
    planned = Counter(row_key(row) for row in old['train'])
    for row in new['rows']:
        rel = Path(row['image'])
        if rel.is_absolute() or '..' in rel.parts:
            raise ValueError('Caminho inseguro no manifesto interno')
        planned[(str(local_dir / rel), int(row['label']), f"cfsmall:{row['group']}")] += 1
    if actual != planned or any(count != 1 for count in actual.values()):
        raise ValueError('Partições não correspondem aos manifestos internos congelados')
    if Counter(row_key(row) for row in split['test_reserved']) != Counter(
            row_key(row) for row in old['test']):
        raise ValueError('Registro do teste histórico reservado mudou')


def summarize(rows, probabilities, threshold):
    """Métricas descritivas com fake=1, sem recalibrar ou escolher limiar."""
    if len(rows) != len(probabilities) or not rows:
        raise ValueError('Predições e rótulos não correspondem')
    if not 0 < threshold < 1:
        raise ValueError('Limiar inválido')
    counts = Counter((int(row['label']), int(prob >= threshold))
                     for row, prob in zip(rows, probabilities))
    tn, fp, fn, tp = (counts[0, 0], counts[0, 1], counts[1, 0], counts[1, 1])
    if min(tn + fp, fn + tp) == 0:
        raise ValueError('Uma classe está ausente da partição')
    recall = tp / (tp + fn)
    specificity = tn / (tn + fp)
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {'n': len(rows), 'real': tn + fp, 'fake': fn + tp,
            'accuracy': (tn + tp) / len(rows), 'balanced_accuracy': (recall + specificity) / 2,
            'precision': precision, 'recall_fake': recall, 'specificity_real': specificity,
            'f1': f1, 'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp}


def group_table(subset, rows, probabilities, threshold):
    groups = defaultdict(list)
    for row, prob in zip(rows, probabilities):
        groups[(int(row['label']), str(row['group']))].append(float(prob))
    report = []
    for (label, group), values in sorted(groups.items()):
        predicted_fake = sum(prob >= threshold for prob in values)
        errors = predicted_fake if label == 0 else len(values) - predicted_fake
        report.append({'subset': subset, 'label': label, 'group': group,
                       'n': len(values), 'errors': errors,
                       'correct_rate': (len(values) - errors) / len(values),
                       'mean_probability_fake': sum(values) / len(values)})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--local', type=Path, default=LOCAL)
    parser.add_argument('--code-root', type=Path, default=CODE_ROOT)
    args = parser.parse_args()
    root, local = args.root, args.local
    run = root / 'modelos/fase3_5_cfsmall_v1'
    output = root / 'resultados/diagnostico_interno_fase3_5_v1'
    source = args.code_root / 'Notebooks/fase_3/fase3_3/fase3.3_compressao.ipynb'
    required = (source, run / 'training_complete.json', run / 'phase3_final.pt',
                run / 'experiment_config.json', run / 'split_manifest.json',
                run / 'internal_metrics.json', root / 'dados_fase3_4/manifest.json',
                local / 'manifest.json')
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    if output.exists():
        summary_path = output / 'summary.json'
        if summary_path.is_file():
            print('Diagnóstico já salvo; resultados preservados:', summary_path, flush=True)
            print(summary_path.read_text(encoding='utf-8'), flush=True)
            return
        raise FileExistsError(f'Diretório parcial existente; não sobrescrever: {output}')

    complete = json.loads((run / 'training_complete.json').read_text(encoding='utf-8'))
    if complete.get('status') != 'internal_training_complete':
        raise ValueError('Treino interno ainda não concluído')
    config = json.loads((run / 'experiment_config.json').read_text(encoding='utf-8'))
    if sha256(source) != config['SOURCE_SHA256']:
        raise ValueError('Fonte da arquitetura difere da execução treinada')
    old_path = root / 'dados_fase3_4/manifest.json'
    new_path = local / 'manifest.json'
    if sha256(old_path) != config['DATA_MANIFEST_SHA256'] or sha256(new_path) != config['NEW_DATA_MANIFEST_SHA256']:
        raise ValueError('Manifesto interno difere do treino concluído')
    split = json.loads((run / 'split_manifest.json').read_text(encoding='utf-8'))
    old = json.loads(old_path.read_text(encoding='utf-8'))
    new = json.loads(new_path.read_text(encoding='utf-8'))
    if split.get('manifest_sha256') != config['DATA_MANIFEST_SHA256'] or split.get('new_manifest_sha256') != config['NEW_DATA_MANIFEST_SHA256']:
        raise ValueError('Registro de partições difere da configuração')
    expected_partitions(split, old, new, local)
    print('Partições internas conferidas; teste reservado não será inferido.', flush=True)

    import torch
    from torch.utils.data import DataLoader
    if not torch.cuda.is_available():
        raise RuntimeError('Conecte uma GPU no Colab')
    notebook = json.loads(source.read_text(encoding='utf-8'))
    scope = {'__name__': '__diagnostic__'}
    for index, prefix in ((3, None), (45, None), (47, 'train_pool ='),
                          (49, '# ---------- carrega o checkpoint final da Fase 2 ----------')):
        code = ''.join(notebook['cells'][index]['source'])
        if prefix:
            if prefix not in code:
                raise RuntimeError(f'Estrutura da célula {index} mudou')
            code = code.split(prefix, 1)[0]
        if index == 3:
            # O código original fixa a GPU; a configuração efetiva vem do treino salvo.
            exec(compile(code, f'{source.name}:cell{index}', 'exec'), scope)
            scope['CFG'] = config
        else:
            exec(compile(code, f'{source.name}:cell{index}', 'exec'), scope)
    checkpoint = torch.load(run / 'phase3_final.pt', map_location='cpu', weights_only=False)
    metrics = json.loads((run / 'internal_metrics.json').read_text(encoding='utf-8'))
    if checkpoint.get('cfg_phase3', {}).get('RUN_NAME') != config['RUN_NAME'] or checkpoint['cfg_phase3'].get('SOURCE_SHA256') != config['SOURCE_SHA256']:
        raise ValueError('Checkpoint final não corresponde à configuração do treino')
    temperature = float(checkpoint['temperature'])
    threshold = float(checkpoint['threshold'])
    if temperature <= 0 or not 0 < threshold < 1:
        raise ValueError('Temperatura ou limiar inválido no checkpoint')
    if abs(temperature - float(metrics['temperature'])) > 1e-8 or abs(threshold - float(metrics['threshold'])) > 1e-8:
        raise ValueError('Calibração do checkpoint difere das métricas registradas')
    cfg2 = checkpoint['cfg']
    model = scope['Phase2Detector'](branches=cfg2['BRANCHES'], fusion=cfg2['FUSION'],
                                    proj_dim=cfg2['PROJ_DIM'], pretrained_cnn=False)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model = model.cuda().eval()
    print(f'Checkpoint final carregado | T={temperature:.6f} | limiar={threshold:.6f}', flush=True)

    all_predictions, all_groups, summaries = [], [], {}
    for subset in DIAGNOSTIC_PARTITIONS:
        rows = split[subset]
        dataset = scope['PathDataset'](rows, scope['multicrop_tf'])
        loader = DataLoader(dataset, batch_size=8, shuffle=False,
                            num_workers=min(2, os.cpu_count() or 1), pin_memory=True)
        probabilities = []
        processed = 0
        with torch.inference_mode():
            for images, _ in loader:
                count = images.shape[0]
                images = images.cuda(non_blocking=True).flatten(0, 1)
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    logits = model(images)
                scores = logits.float().view(count, 5, 2).mean(1)
                if not torch.isfinite(scores).all():
                    raise ValueError(f'Logits não finitos na partição {subset}')
                probabilities.extend(torch.softmax(scores / temperature, 1)[:, 1].cpu().tolist())
                processed += count
                if processed // 100 != (processed - count) // 100 or processed == len(rows):
                    print(f'{subset}: {processed}/{len(rows)} inferências', flush=True)
        if len(probabilities) != len(rows):
            raise ValueError('Quantidade de predições incompleta')
        summary = summarize(rows, probabilities, threshold)
        from sklearn.metrics import roc_auc_score
        summary['auc'] = float(roc_auc_score([int(row['label']) for row in rows], probabilities))
        summary['protocol'] = 'five_crop; frozen_temperature_and_threshold'
        summary['interpretation'] = 'model_selected_on_this_subset; descriptive_only'
        summaries[subset] = summary
        all_groups.extend(group_table(subset, rows, probabilities, threshold))
        for row, prob in zip(rows, probabilities):
            prediction = int(prob >= threshold)
            all_predictions.append({'subset': subset, 'path': row['path'],
                                    'label': int(row['label']), 'group': row['group'],
                                    'probability_fake': prob, 'prediction': prediction,
                                    'correct': int(prediction == int(row['label']))})
        print(f"{subset}: acurácia={summary['accuracy']:.4f} | recall_fake={summary['recall_fake']:.4f} | "
              f"especificidade_real={summary['specificity_real']:.4f} | AUC={summary['auc']:.4f}", flush=True)

    for label, name in ((0, 'reais'), (1, 'sintéticas')):
        eligible = sorted((item for item in all_groups if item['label'] == label and item['n'] >= 10),
                          key=lambda item: (item['correct_rate'], -item['n'], item['group']))
        print(f'Grupos {name} com 10+ imagens; até 10 menores taxas de acerto:', flush=True)
        for item in eligible[:10]:
            print(f"  {item['group']} | n={item['n']} | erros={item['errors']} | "
                  f"acerto={item['correct_rate']:.3f}", flush=True)

    output.mkdir(parents=True, exist_ok=False)
    summary = {'checkpoint': str(run / 'phase3_final.pt'),
               'checkpoint_sha256': sha256(run / 'phase3_final.pt'),
               'split_manifest_sha256': sha256(run / 'split_manifest.json'),
               'source_sha256': sha256(source),
               'temperature_frozen': temperature, 'threshold_frozen': threshold,
               'external_used': False, 'openfake_test_inferred': False,
               'calibration_inferred': False,
               'note': 'Somente diagnóstico na seleção interna; esta partição escolheu o modelo. Calibração, OpenFake teste e benchmark externo não são inferidos. Cinco recortes podem produzir AUC diferente da AUC central registrada durante o treino. Fontes reais CF não foram informadas.',
               'partitions': summaries}
    with (output / 'predictions.csv').open('x', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=all_predictions[0].keys())
        writer.writeheader(); writer.writerows(all_predictions)
    with (output / 'per_group.csv').open('x', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=all_groups[0].keys())
        writer.writeheader(); writer.writerows(all_groups)
    with (output / 'summary.json').open('x', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print('Diagnóstico interno salvo:', output, flush=True)


if __name__ == '__main__':
    main()
