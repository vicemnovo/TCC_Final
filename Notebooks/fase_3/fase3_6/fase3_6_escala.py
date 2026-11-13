"""Fase 3.6: teste isolado de robustez a redimensionamento, sem teste externo.

Reutiliza os dados, divisões, pesos iniciais e treinador da Fase 3.5.
Única mudança experimental: RandomDownUp de 0,15 para 0,50 no treino.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import sys
for _module_dir in (
    Path(__file__).resolve().parents[3] / 'experimentos',
):
    sys.path.insert(0, str(_module_dir))

from amostragem_fase3_4 import balanced_sqrt_group_weights
from particoes_fase3_4 import split_payload, persist_or_verify

ROOT = Path('/content/drive/MyDrive/TCC_Final')
CODE_ROOT = Path(os.environ.get('TCC_CODE_ROOT', '/content/tcc_codigo/ausente'))
SOURCE = CODE_ROOT / 'Notebooks/fase_3/fase3_3/fase3.3_compressao.ipynb'
RUN_NAME = 'fase3_6_escala_p05_v1'
RUN_DIR = ROOT / 'modelos' / RUN_NAME
START = ROOT / 'modelos/fase3_3_compressao_v1/phase3_final.pt'
OLD_MANIFEST = ROOT / 'dados_fase3_4/manifest.json'
NEW_LOCAL = Path('/content/fase3_5_novos_v1')
NEW_MANIFEST = NEW_LOCAL / 'manifest.json'

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('mode', choices=('smoke', 'train', 'resume', 'auto'))
args = parser.parse_args()

import torch
if not ROOT.is_dir():
    raise RuntimeError('Monte o Drive no notebook antes de executar')
for required in (SOURCE, START, OLD_MANIFEST, NEW_MANIFEST):
    if not required.is_file():
        raise FileNotFoundError(f'Arquivo necessário ausente: {required}')

recover_initial = False
if args.mode == 'auto':
    if (RUN_DIR / 'training_complete.json').is_file():
        print('Experimento interno já concluído:', RUN_DIR)
        raise SystemExit(0)
    if (RUN_DIR / 'last.pt').is_file():
        args.mode = 'resume'
    else:
        args.mode = 'train'
        recover_initial = RUN_DIR.is_dir() and (RUN_DIR / 'experiment_config.json').is_file() and not any(
            (RUN_DIR / name).exists() for name in
            ('best.pt', 'best.pt.tmp', 'last.pt.tmp', 'history.csv',
             'phase3_final.pt', 'training_complete.json')
        )
if args.mode == 'train' and RUN_DIR.exists() and not recover_initial:
    raise FileExistsError(f'Experimento já existe; não sobrescrever: {RUN_DIR}')
if args.mode == 'resume' and not (RUN_DIR / 'last.pt').is_file():
    raise FileNotFoundError(f'Checkpoint de retomada ausente: {RUN_DIR / "last.pt"}')
if args.mode == 'resume' and (RUN_DIR / 'training_complete.json').is_file():
    raise FileExistsError('Treinamento interno já finalizado')

nb = json.loads(SOURCE.read_text(encoding='utf-8'))
def cell(index, prefix=None):
    src = ''.join(nb['cells'][index]['source'])
    if index == 49:
        old_message = 'Fase 2 carregada |'
        if old_message not in src:
            raise RuntimeError('Texto de montagem da arquitetura mudou')
        src = src.replace(old_message, 'Arquitetura da Fase 2 montada |', 1)
    if prefix:
        if prefix not in src:
            raise RuntimeError(f'Marcador ausente na célula {index}: {prefix}')
        src = src.split(prefix, 1)[0]
    exec(compile(src, f'{SOURCE.name}:cell{index}', 'exec'), globals())

cell(3)
CFG.update(RUN_NAME=RUN_NAME, EPOCHS=12, PATIENCE=3,
           LR_VIT=5e-6, LR_FFT=1e-5, LR_CNN=1e-5, LR_HEAD=3e-5,
           LABEL_SMOOTHING=0.05, BATCH_SIZE=32,
           REPLAY_DIR='', REPLAY_PER_CLASS=0,
           SAMPLER_POLICY='class_balanced_sqrt_group_v1')
CFG['SCALE_AUGMENT_P'] = 0.50
CFG['START_CKPT'] = str(START)
CFG['DATA_MANIFEST'] = str(OLD_MANIFEST)
CFG['NEW_DATA_MANIFEST'] = str(NEW_MANIFEST)
CFG['SOURCE_SHA256'] = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
CFG['SCRIPT_SHA256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
CFG['DATA_MANIFEST_SHA256'] = hashlib.sha256(OLD_MANIFEST.read_bytes()).hexdigest()
CFG['NEW_DATA_MANIFEST_SHA256'] = hashlib.sha256(NEW_MANIFEST.read_bytes()).hexdigest()

cell(11)
cell(45)
# Os demais aumentos são os da 3.5; só muda a frequência do redimensionamento.
train_tf = transforms.Compose([
    transforms.RandomCrop(CFG['CROP'], pad_if_needed=True),
    transforms.RandomHorizontalFlip(), RandomBlur(0.1), RandomDownUp(CFG['SCALE_AUGMENT_P']),
    RandomJPEG(0.5, (55, 95)), _to_norm,
])
cell(47, 'train_pool =')
cell(49)
cell(51)

old = json.loads(OLD_MANIFEST.read_text(encoding='utf-8'))
new = json.loads(NEW_MANIFEST.read_text(encoding='utf-8'))
if new.get('name') != 'fase3_5_novos_v1' or new.get('source') != 'OwensLab/CommunityForensics-Small':
    raise ValueError('Manifesto novo não é a amostra interna planejada')
old_pool = old['train']
test_rows = old['test']  # Somente auditoria de separação; NUNCA para avaliação da 3.6.
tr_old, ves_old, vcal_old = split_by_group(old_pool, CFG['VAL_FRAC'], CFG['SEED'])
new_by_split = {'train': [], 'validation_selection': [], 'validation_calibration': []}
for row in new['rows']:
    subset = row['subset']
    if subset not in new_by_split:
        raise ValueError(f'Divisão nova inválida: {subset}')
    rel = Path(row['image'])
    if rel.is_absolute() or '..' in rel.parts:
        raise ValueError('Caminho inseguro no manifesto novo')
    path = NEW_LOCAL / rel
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != row['image_sha256']:
        raise ValueError(f'Imagem nova ausente ou alterada: {rel}')
    new_by_split[subset].append({'path': str(path), 'label': int(row['label']),
                                 'group': f"cfsmall:{row['group']}"})

tr_rows = tr_old + new_by_split['train']
ves_rows = ves_old + new_by_split['validation_selection']
vcal_rows = vcal_old + new_by_split['validation_calibration']
for name, rows in (('treino', tr_rows), ('seleção', ves_rows), ('calibração', vcal_rows)):
    if {int(row['label']) for row in rows} != {0, 1}:
        raise ValueError(f'Falta classe em {name}')
print('Partições 3.6:', len(tr_rows), len(ves_rows), len(vcal_rows),
      '| teste histórico reservado, não avaliado:', len(test_rows), flush=True)

def digests(rows, title, *, reject_internal=True):
    result = set()
    for index, row in enumerate(rows, 1):
        path = Path(row['path'])
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open('rb') as stream:
            value = hashlib.file_digest(stream, 'sha256').digest()
        if reject_internal and value in result:
            raise ValueError(f'Duplicata exata dentro de {title}')
        result.add(value)
        if index % 500 == 0:
            print(f'Hashes {title}: {index}/{len(rows)}', flush=True)
    return result

hash_sets = [digests(rows, title) for rows, title in
             ((tr_rows, 'treino'), (ves_rows, 'seleção'),
              (vcal_rows, 'calibração'))]
# O teste histórico OpenFake contém três pares internos já auditados; não o
# avaliamos aqui, mas ainda conferimos que seus hashes não vazam para o treino.
hash_sets.append(digests(test_rows, 'teste reservado', reject_internal=False))
for i in range(4):
    for j in range(i + 1, 4):
        if hash_sets[i] & hash_sets[j]:
            raise ValueError(f'Conteúdo duplicado entre partições {i} e {j}')

benchmark_path = ROOT / 'dados_externos/benchmark_externo_raise_mnw_2000_v1/manifest.json'
if not benchmark_path.is_file():
    raise FileNotFoundError(benchmark_path)
benchmark_rows = json.loads(benchmark_path.read_text(encoding='utf-8'))['rows']
benchmark_hashes = {bytes.fromhex(row['image_sha256']) for row in benchmark_rows}
if benchmark_hashes & (hash_sets[0] | hash_sets[1] | hash_sets[2]):
    raise ValueError('Imagem do benchmark externo encontrada em dados internos')

partition_record = split_payload(
    tr_rows, ves_rows, vcal_rows, test_rows,
    manifest_sha256=CFG['DATA_MANIFEST_SHA256'],
    replay_count=0, sampler_policy=CFG['SAMPLER_POLICY'])
partition_record['new_manifest_sha256'] = CFG['NEW_DATA_MANIFEST_SHA256']
baseline_split_path = ROOT / 'modelos/fase3_5_cfsmall_v1/split_manifest.json'
if not baseline_split_path.is_file() or json.loads(baseline_split_path.read_text(encoding='utf-8')) != partition_record:
    raise ValueError('Partições da 3.6 diferem das da 3.5; comparação recusada')
print('Partições idênticas às da Fase 3.5 confirmadas.', flush=True)

from torch.utils.data import DataLoader, WeightedRandomSampler
nw = CFG['NUM_WORKERS']
weights = balanced_sqrt_group_weights(tr_rows)
train_loader = DataLoader(PathDataset(tr_rows, train_tf), batch_size=CFG['BATCH_SIZE'],
    sampler=WeightedRandomSampler(weights, len(tr_rows), replacement=True),
    num_workers=nw, pin_memory=True, drop_last=True, persistent_workers=nw > 0)
ves_loader = DataLoader(PathDataset(ves_rows, eval_tf), batch_size=64,
                        shuffle=False, num_workers=nw)
vcal_loader = DataLoader(PathDataset(vcal_rows, multicrop_tf), batch_size=8,
                         shuffle=False, num_workers=nw)

start = torch.load(START, map_location='cpu', weights_only=False)
model.load_state_dict(start['model_state_dict'], strict=True)
print('Pesos iniciais da Fase 3.3 carregados:', START, flush=True)
del start

if args.mode == 'smoke':
    x, _ = next(iter(DataLoader(PathDataset(tr_rows[:1], eval_tf), batch_size=1)))
    model.eval()
    with torch.no_grad():
        logits = model(x.to(device)).float().cpu()
    if logits.shape != (1, 2) or not torch.isfinite(logits).all():
        raise ValueError('Forward não finito ou forma inválida')
    smoke_dir = ROOT / 'modelos' / (RUN_NAME + '_smoke')
    smoke_dir.mkdir(parents=True, exist_ok=True)
    smoke_path = smoke_dir / 'smoke.json'
    if smoke_path.exists():
        raise FileExistsError(f'Smoke anterior preservado: {smoke_path}')
    persist_or_verify(smoke_dir / 'split_manifest.json', partition_record)
    with smoke_path.open('x', encoding='utf-8') as stream:
        json.dump({'checkpoint': str(START), 'forward_finite': True,
                   'logits_shape': list(logits.shape), 'train_count': len(tr_rows),
                   'selection_count': len(ves_rows), 'calibration_count': len(vcal_rows),
                   'external_used': False}, stream, indent=2)
    print('Smoke OK:', smoke_path, flush=True)
    raise SystemExit(0)

if args.mode == 'train' and not recover_initial:
    RUN_DIR.mkdir(parents=True, exist_ok=False)
for dirname in ('checkpoints', 'logs', 'historicos', 'resultados'):
    (RUN_DIR / dirname).mkdir(parents=True, exist_ok=True)
LAST_CKPT = str(RUN_DIR / 'last.pt')
BEST_CKPT = str(RUN_DIR / 'best.pt')
HISTORY_CSV = str(RUN_DIR / 'history.csv')
config_path = RUN_DIR / 'experiment_config.json'
if args.mode == 'resume' or recover_initial:
    if json.loads(config_path.read_text(encoding='utf-8')) != json.loads(json.dumps(CFG)):
        raise ValueError('Configuração alterada; retomada recusada')
else:
    with config_path.open('x', encoding='utf-8') as stream:
        json.dump(CFG, stream, indent=2)
persist_or_verify(RUN_DIR / 'split_manifest.json', partition_record,
                  resume=args.mode == 'resume')

train_source = ''.join(nb['cells'][55]['source'])
if not train_source.startswith('RESUME = False'):
    raise RuntimeError('Treinador histórico mudou')
train_source = train_source.replace('RESUME = False', f'RESUME = {args.mode == "resume"}', 1)
exec(compile(train_source, f'{SOURCE.name}:cell55', 'exec'), globals())

# A seleção interna decide o checkpoint; a calibração interna ajusta T e limiar.
# Nem OpenFake teste nem benchmark externo são inferidos neste experimento.
lg_cal, y_cal_t = collect_tta(model, vcal_loader)
T = fit_temperature(lg_cal, y_cal_t)
p_cal = F.softmax(lg_cal / T, 1)[:, 1].numpy()
thr, bal_cal = pick_threshold(p_cal, y_cal_t.numpy())
metrics = {'best_epoch': best['epoch'], 'selection_auc': best['val']['auc'],
           'calibration_bal_acc_apparent': bal_cal, 'temperature': T,
           'threshold': thr, 'train_count': len(tr_rows),
           'selection_count': len(ves_rows), 'calibration_count': len(vcal_rows),
           'external_used': False, 'openfake_test_used': False}
torch.save({'model_state_dict': model.state_dict(), 'temperature': T,
            'threshold': thr, 'cfg': cfg2, 'cfg_phase3': CFG,
            'preproc': 'JPEG q90 novos; 5 recortes 224 na calibração'},
           RUN_DIR / 'phase3_final.pt')
with (RUN_DIR / 'internal_metrics.json').open('x', encoding='utf-8') as stream:
    json.dump(metrics, stream, indent=2)
with (RUN_DIR / 'training_complete.json').open('x', encoding='utf-8') as stream:
    json.dump({'status': 'internal_training_complete',
               'checkpoint': 'phase3_final.pt', 'external_evaluation': 'not_run'},
              stream, indent=2)
print(json.dumps(metrics, indent=2), flush=True)
print('Treino interno concluído sem avaliação externa:', RUN_DIR, flush=True)
