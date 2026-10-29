"""Audita erros da seleção interna já inferida; não treina nem acessa testes."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path('/content/drive/MyDrive/TCC_Final')
LOCAL = Path('/content/fase3_5_novos_v1')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def size_band(width, height):
    short = min(width, height)
    if short < 512:
        return 'lado_curto_menor_512'
    if short < 1024:
        return 'lado_curto_512_a_1023'
    return 'lado_curto_1024_ou_mais'


def aspect_band(width, height):
    ratio = width / height
    if ratio < 0.8:
        return 'vertical'
    if ratio > 1.25:
        return 'horizontal'
    return 'quase_quadrada'


def validate_predictions(predictions, selection, threshold):
    if len(predictions) != len(selection):
        raise ValueError('Predições não correspondem à seleção congelada')
    seen = set()
    for prediction, row in zip(predictions, selection):
        key = (prediction['path'], int(prediction['label']), prediction['group'])
        expected = (str(row['path']), int(row['label']), str(row['group']))
        if key != expected or key in seen or prediction['subset'] != 'validation_selection':
            raise ValueError('Ordem, rótulo, grupo ou divisão das predições mudou')
        seen.add(key)
        probability = float(prediction['probability_fake'])
        if not 0 <= probability <= 1 or probability != probability:
            raise ValueError('Probabilidade inválida')
        predicted = int(probability >= threshold)
        if int(prediction['prediction']) != predicted or int(prediction['correct']) != int(predicted == key[1]):
            raise ValueError('Rótulo previsto difere do limiar congelado')


def aggregate(records, field):
    groups = defaultdict(list)
    for record in records:
        groups[str(record[field])].append(record)
    result = []
    for value, members in sorted(groups.items()):
        fp = sum(row['error_type'] == 'FP' for row in members)
        fn = sum(row['error_type'] == 'FN' for row in members)
        result.append({'dimension': field, 'value': value, 'n': len(members),
                       'errors': fp + fn, 'error_rate': (fp + fn) / len(members),
                       'false_positives': fp, 'false_negatives': fn})
    return result


def audit(root, local):
    from PIL import Image

    run = root / 'modelos/fase3_5_cfsmall_v1'
    diagnostic = root / 'resultados/diagnostico_interno_fase3_5_v1'
    output = root / 'resultados/auditoria_erros_internos_fase3_5_v1'
    required = (run / 'phase3_final.pt', run / 'split_manifest.json',
                diagnostic / 'summary.json', diagnostic / 'predictions.csv',
                root / 'dados_fase3_4/manifest.json', local / 'manifest.json')
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    if output.exists():
        report = output / 'summary.json'
        if report.is_file():
            print('Auditoria interna já salva; resultados preservados:', report, flush=True)
            print(report.read_text(encoding='utf-8'), flush=True)
            return
        raise FileExistsError(f'Diretório parcial existente; não sobrescrever: {output}')

    prior = json.loads((diagnostic / 'summary.json').read_text(encoding='utf-8'))
    if (prior.get('external_used') is not False or prior.get('calibration_inferred') is not False
            or prior.get('openfake_test_inferred') is not False):
        raise ValueError('Relatório de origem não é restrito à seleção interna')
    if sha256(run / 'phase3_final.pt') != prior['checkpoint_sha256']:
        raise ValueError('Checkpoint alterado desde o diagnóstico')
    split_path = run / 'split_manifest.json'
    if sha256(split_path) != prior['split_manifest_sha256']:
        raise ValueError('Partições alteradas desde o diagnóstico')
    split = json.loads(split_path.read_text(encoding='utf-8'))
    selection = split['validation_selection']
    with (diagnostic / 'predictions.csv').open(newline='', encoding='utf-8') as handle:
        predictions = list(csv.DictReader(handle))
    threshold = float(prior['threshold_frozen'])
    validate_predictions(predictions, selection, threshold)

    old = json.loads((root / 'dados_fase3_4/manifest.json').read_text(encoding='utf-8'))
    new = json.loads((local / 'manifest.json').read_text(encoding='utf-8'))
    old_meta = {str(row['path']): row for row in old['train']}
    new_meta = {str(local / row['image']): row for row in new['rows']}
    if len(old_meta) != len(old['train']) or len(new_meta) != len(new['rows']):
        raise ValueError('Caminho repetido em manifesto interno')

    records = []
    for index, row in enumerate(predictions, 1):
        path = Path(row['path'])
        if row['path'] in old_meta:
            meta = old_meta[row['path']]
            dataset = 'OpenFake'
            original_format = str(meta.get('orig_format') or 'não_registrado').upper()
            original_width = meta.get('orig_w')
            original_height = meta.get('orig_h')
        elif row['path'] in new_meta:
            meta = new_meta[row['path']]
            dataset = 'CommunityForensics-Small'
            original_format = 'não_registrado'
            original_width = original_height = None
        else:
            raise ValueError(f'Imagem da seleção ausente dos manifestos internos: {path}')
        if int(row['label']) != int(meta['label']):
            raise ValueError(f'Rótulo difere do manifesto interno: {path}')
        with Image.open(path) as image:
            width, height = image.size
            processed_format = image.format
        if processed_format != 'JPEG':
            raise ValueError(f'Imagem processada não está em JPEG: {path}')
        label = int(row['label'])
        prediction = int(row['prediction'])
        probability = float(row['probability_fake'])
        error_type = ('FP' if label == 0 else 'FN') if prediction != label else ''
        records.append({'path': str(path), 'dataset': dataset, 'label': label,
                        'group': row['group'], 'probability_fake': probability,
                        'prediction': prediction, 'error_type': error_type,
                        'wrong_label_probability': (probability if label == 0 else 1 - probability)
                        if error_type else '',
                        'original_format': original_format,
                        'original_width': original_width or '',
                        'original_height': original_height or '',
                        'processed_width': width, 'processed_height': height,
                        'processed_short_side': min(width, height),
                        'processed_size_band': size_band(width, height),
                        'processed_aspect_band': aspect_band(width, height)})
        if index % 100 == 0 or index == len(predictions):
            print(f'Metadados internos: {index}/{len(predictions)}', flush=True)

    errors = [row for row in records if row['error_type']]
    counts = Counter(row['error_type'] for row in errors)
    expected = prior['partitions']['validation_selection']
    if (len(records) != expected['n'] or counts['FP'] != expected['fp']
            or counts['FN'] != expected['fn']):
        raise ValueError('Contagem de erros difere do diagnóstico congelado')
    dimensions = ('dataset', 'label', 'group', 'original_format',
                  'processed_size_band', 'processed_aspect_band')
    segments = [entry for dimension in dimensions for entry in aggregate(records, dimension)]
    print(f"Seleção: {len(records)} imagens; FP={counts['FP']}; FN={counts['FN']}", flush=True)
    for dimension in ('dataset', 'original_format', 'processed_size_band'):
        print(dimension + ':', flush=True)
        for entry in (item for item in segments if item['dimension'] == dimension):
            print(f"  {entry['value']}: {entry['errors']}/{entry['n']} erros "
                  f"(FP={entry['false_positives']}, FN={entry['false_negatives']})", flush=True)
    most_confident = sorted(errors, key=lambda item: item['wrong_label_probability'], reverse=True)[:12]
    print('Erros mais confiantes (probabilidade da classe incorreta):', flush=True)
    for item in most_confident:
        print(f"  {item['error_type']} | {item['dataset']} | {item['group']} | "
              f"p_fake={item['probability_fake']:.3f} | {Path(item['path']).name}", flush=True)

    output.mkdir(parents=True, exist_ok=False)
    with (output / 'selection_metadata.csv').open('x', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys())
        writer.writeheader(); writer.writerows(records)
    with (output / 'errors.csv').open('x', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys())
        writer.writeheader(); writer.writerows(errors)
    with (output / 'segments.csv').open('x', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=segments[0].keys())
        writer.writeheader(); writer.writerows(segments)
    summary = {'selection_count': len(records), 'false_positives': counts['FP'],
               'false_negatives': counts['FN'], 'checkpoint_sha256': prior['checkpoint_sha256'],
               'diagnostic_summary_sha256': sha256(diagnostic / 'summary.json'),
               'predictions_sha256': sha256(diagnostic / 'predictions.csv'),
               'external_used': False, 'calibration_used': False,
               'note': 'Associações descritivas na seleção que escolheu o modelo; não provam causalidade nem generalização externa.'}
    with (output / 'summary.json').open('x', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print('Auditoria salva:', output, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--local', type=Path, default=LOCAL)
    args = parser.parse_args()
    audit(args.root, args.local)


if __name__ == '__main__':
    main()
