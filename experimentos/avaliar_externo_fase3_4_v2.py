"""Avaliação única das 20 imagens externas com a Fase 3.4 v2 congelada.

Não treina, não calibra e não escolhe limiar. Os resultados são gravados em
uma pasta exclusiva; uma execução existente nunca é substituída.
"""

import argparse
import collections
import csv
import hashlib
import io
import json
import math
import os
import random
import time
from pathlib import Path


ROOT_DEFAULT = Path('/content/drive/MyDrive/TCC_Final')
RUN_NAME = 'fase3_4_finetune_v2_sqrt_sampler'
EVAL_NAME = 'avaliacao_externa_fase3_4_v2_v1'
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}


def sha256_file(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def load_external(root):
    """Lê apenas o conjunto rotulado; não infere rótulos por nomes/pastas."""
    external_dir = root / 'validacao_fase3' / 'teste_externo'
    labels_path = external_dir / 'rotulos.csv'
    if not labels_path.is_file():
        raise FileNotFoundError(f'Rótulos externos ausentes: {labels_path}')
    with labels_path.open(newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        if not {'arquivo', 'label_real'}.issubset(reader.fieldnames or []):
            raise ValueError('rotulos.csv precisa das colunas arquivo e label_real')
        raw_rows = list(reader)
    if len(raw_rows) != 20:
        raise ValueError(f'Esperadas 20 imagens externas; encontradas {len(raw_rows)} linhas')
    rows, names, hashes = [], set(), set()
    for row in raw_rows:
        name = row['arquivo'].strip()
        if not name or Path(name).name != name or name.casefold() in names:
            raise ValueError(f'Nome ausente, repetido ou inseguro no CSV: {name!r}')
        if Path(name).suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f'Extensão não suportada: {name!r}')
        if row['label_real'] not in ('0', '1'):
            raise ValueError(f'Rótulo inválido para {name!r}')
        path = external_dir / name
        if not path.is_file():
            raise FileNotFoundError(f'Imagem externa ausente: {path}')
        digest = sha256_file(path)
        if digest in hashes:
            raise ValueError(f'Conteúdo duplicado no conjunto externo: {name!r}')
        names.add(name.casefold())
        hashes.add(digest)
        rows.append({'path': str(path), 'name': name,
                     'label': int(row['label_real']), 'sha256': digest})
    counts = collections.Counter(row['label'] for row in rows)
    if counts != {0: 8, 1: 12}:
        raise ValueError(f'Distribuição externa alterada; esperado 8 reais e 12 sintéticas: {counts}')
    return labels_path, rows


def verify_no_openfake_overlap(root, external_rows, progress_every=100, expected_count=3979):
    """Compara conteúdo às quatro partições registradas, inclusive teste."""
    run_dir = root / 'modelos' / RUN_NAME
    manifest = root / 'dados_fase3_4' / 'manifest.json'
    split_path = run_dir / 'split_manifest.json'
    if not manifest.is_file() or not split_path.is_file():
        raise FileNotFoundError('Manifesto OpenFake ou partições v2 ausentes')
    splits = json.loads(split_path.read_text(encoding='utf-8'))
    if sha256_file(manifest) != splits['manifest_sha256']:
        raise ValueError('O manifesto OpenFake mudou desde o treino v2')
    keys = ('train', 'validation_selection', 'validation_calibration', 'test_reserved')
    paths = [Path(row['path']) for key in keys for row in splits[key]]
    if len(paths) != expected_count:
        raise ValueError(f'Tamanho das partições v2 mudou: {len(paths)} em vez de {expected_count}')
    external_hashes = {row['sha256'] for row in external_rows}
    started = time.monotonic()
    print(f'Conferindo sobreposição exata com {len(paths)} imagens OpenFake...', flush=True)
    for index, path in enumerate(paths, 1):
        if not path.is_file():
            raise FileNotFoundError(f'Imagem OpenFake ausente: {path}')
        if sha256_file(path) in external_hashes:
            raise ValueError(f'Imagem externa encontrada nas partições OpenFake: {path}')
        if index % progress_every == 0 or index == len(paths):
            print(f'Integridade: {index}/{len(paths)} arquivos verificados '
                  f'({(time.monotonic() - started) / 60:.1f} min)', flush=True)
    return sha256_file(split_path)


def evaluate(root, code_root):
    # O resultado final é bloqueado antes de abrir qualquer imagem para inferência.
    out_dir = root / 'resultados' / EVAL_NAME
    if out_dir.exists():
        raise FileExistsError(f'Avaliação externa anterior preservada: {out_dir}')
    run_dir = root / 'modelos' / RUN_NAME
    checkpoint_path = run_dir / 'phase3_final.pt'
    metrics_path = run_dir / 'final_metrics.json'
    config_path = run_dir / 'experiment_config.json'
    source = code_root / 'Notebooks' / 'fase_3' / 'fase3.3_compressao.ipynb'
    required = (checkpoint_path, metrics_path, config_path, source)
    if not all(path.is_file() for path in required):
        raise FileNotFoundError(f'Artefatos v2/código ausentes: {[str(p) for p in required if not p.is_file()]}')
    config = json.loads(config_path.read_text(encoding='utf-8'))
    if sha256_file(source) != config['SOURCE_SHA256']:
        raise ValueError('Notebook de arquitetura difere daquele registrado no treino v2')

    labels_path, external_rows = load_external(root)
    split_sha = verify_no_openfake_overlap(root, external_rows)
    print('Conjunto externo: 20 arquivos únicos; 8 reais e 12 sintéticos; '
          'sem sobreposição exata com as partições v2.', flush=True)

    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from PIL import Image, ImageFilter
    from sklearn.metrics import confusion_matrix, roc_auc_score
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

    if not torch.cuda.is_available():
        raise RuntimeError('Conecte uma GPU do Colab para a avaliação externa')
    device = torch.device('cuda')
    print('GPU:', torch.cuda.get_device_name(0), flush=True)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    saved_metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
    temperature = float(checkpoint['temperature'])
    threshold = float(checkpoint['threshold'])
    if not (math.isclose(temperature, saved_metrics['temperature'], abs_tol=1e-8)
            and math.isclose(threshold, saved_metrics['threshold'], abs_tol=1e-8)):
        raise ValueError('Temperatura ou limiar divergem entre checkpoint e métricas v2')
    if checkpoint['cfg_phase3']['RUN_NAME'] != RUN_NAME:
        raise ValueError('Checkpoint não pertence à execução v2 esperada')
    CFG = checkpoint['cfg_phase3']
    notebook = json.loads(source.read_text(encoding='utf-8'))
    namespace = {'CFG': CFG, 'device': device, 'np': np, 'torch': torch,
                 'nn': nn, 'F': F, 'transforms': transforms,
                 'Image': Image, 'ImageFilter': ImageFilter,
                 'io': io, 'random': random, 'collections': collections,
                 'math': math, 'csv': csv, 'os': os, 'time': time,
                 'confusion_matrix': confusion_matrix, 'roc_auc_score': roc_auc_score,
                 'efficientnet_b0': efficientnet_b0,
                 'EfficientNet_B0_Weights': EfficientNet_B0_Weights}
    for index, marker in ((45, None), (49, '# ---------- carrega o checkpoint final da Fase 2 ----------'), (51, None)):
        code = ''.join(notebook['cells'][index]['source'])
        if marker:
            if marker not in code:
                raise RuntimeError(f'Marcador da arquitetura ausente na célula {index}')
            code = code.split(marker, 1)[0]
        exec(compile(code, f'{source.name}:cell{index}', 'exec'), namespace)
    arch = checkpoint['cfg']
    model = namespace['Phase2Detector'](
        branches=arch['BRANCHES'], fusion=arch['FUSION'],
        proj_dim=arch['PROJ_DIM'], pretrained_cnn=False)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model = model.to(device).eval()
    print(f'Pesos finais da Fase 3.4 v2 carregados | T={temperature:.6f} '
          f'| limiar={threshold:.6f}', flush=True)

    class ExternalDataset(Dataset):
        def __len__(self):
            return len(external_rows)

        def __getitem__(self, index):
            row = external_rows[index]
            with Image.open(row['path']) as image:
                image = image.convert('RGB')
                if min(image.size) < CFG['CROP']:
                    raise ValueError(f'Imagem menor que {CFG["CROP"]} px: {row["name"]}')
                return namespace['multicrop_tf'](image), row['label']

    loader = DataLoader(ExternalDataset(), batch_size=4, shuffle=False, num_workers=0)
    print('Inferência final: 20 imagens, cinco recortes por imagem...', flush=True)
    logits, labels = namespace['collect_tta'](model, loader)
    probabilities = F.softmax(logits / temperature, dim=1)[:, 1].numpy()
    y_true = labels.numpy()
    y_pred = (probabilities >= threshold).astype(int)
    result = namespace['binary_metrics'](y_true, y_pred)
    result['auc'] = float(roc_auc_score(y_true, probabilities))
    result['accuracy_ci95_wilson'] = list(namespace['wilson_ci'](int((y_true == y_pred).sum()), len(y_true)))
    result.update({'n': len(y_true), 'real': 8, 'fake': 12,
                   'temperature_frozen': temperature, 'threshold_frozen': threshold,
                   'checkpoint': str(checkpoint_path), 'checkpoint_sha256': sha256_file(checkpoint_path),
                   'labels_sha256': sha256_file(labels_path), 'split_manifest_sha256': split_sha,
                   'source_sha256': sha256_file(source), 'evaluation_script_sha256': sha256_file(Path(__file__)),
                   'provenance_note': 'Origem das 20 imagens está como desconhecido em rotulos.csv; '
                                      'ausência de duplicatas exatas não comprova independência de origem.'})
    predictions = [
        {'arquivo': row['name'], 'label_real': row['label'], 'sha256': row['sha256'],
         'prob_fake': float(prob), 'pred_limiar': int(pred), 'acerto': int(pred == row['label'])}
        for row, prob, pred in zip(external_rows, probabilities, y_pred)
    ]
    out_dir.mkdir(parents=True, exist_ok=False)
    with (out_dir / 'predictions.csv').open('x', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=predictions[0].keys())
        writer.writeheader()
        writer.writerows(predictions)
    with (out_dir / 'metrics.json').open('x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    print('Avaliação externa salva, sem alterar o experimento:', out_dir, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT_DEFAULT)
    parser.add_argument('--code-root', type=Path, default=Path(os.environ.get('TCC_CODE_ROOT', '/content/tcc_codigo/ausente')))
    args = parser.parse_args()
    evaluate(args.root, args.code_root)


if __name__ == '__main__':
    main()
