"""Ajuste fino da Fase 3.4; código sincronizado do GitHub, artefatos no Drive."""

import csv
import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path('/content/drive/MyDrive/TCC_Final')
CODE_ROOT = Path(os.environ.get('TCC_CODE_ROOT', '/content/tcc_codigo/ausente'))
SOURCE = CODE_ROOT / 'Notebooks/fase3 - openfake/fase3.3_compressao.ipynb'
RUN_NAME = 'fase3_4_finetune_v1'
RUN_DIR = ROOT / 'modelos' / RUN_NAME
START = ROOT / 'modelos/fase3_3_compressao_v1/phase3_final.pt'
DATA_DIR = ROOT / 'dados_fase3_4'
MANIFEST_PATH = DATA_DIR / 'manifest.json'

parser = argparse.ArgumentParser()
parser.add_argument('mode', choices=('prepare', 'smoke', 'train', 'resume', 'auto'))
args = parser.parse_args()

import torch
if not ROOT.is_dir():
    raise RuntimeError(f'Monte o Google Drive no notebook antes de executar: {ROOT}')
if not SOURCE.is_file() or not START.is_file():
    raise FileNotFoundError('Necessários: código sincronizado e checkpoint 3.3 no Drive.')
if args.mode == 'auto':
    if (RUN_DIR / 'final_metrics.json').exists():
        print('Experimento já concluído:', RUN_DIR)
        raise SystemExit(0)
    args.mode = 'resume' if (RUN_DIR / 'last.pt').is_file() else 'train'
if args.mode == 'train' and RUN_DIR.exists():
    raise FileExistsError(f'Experimento já existe; use resume ou altere RUN_NAME: {RUN_DIR}')
if args.mode == 'resume' and not (RUN_DIR / 'last.pt').is_file():
    raise FileNotFoundError(f'Checkpoint de retomada ausente: {RUN_DIR / "last.pt"}')
if args.mode == 'resume' and (RUN_DIR / 'final_metrics.json').exists():
    raise FileExistsError('Esta execução já terminou; resultados finais preservados.')

nb = json.loads(SOURCE.read_text(encoding='utf-8'))
def cell(index, prefix=None):
    src = ''.join(nb['cells'][index]['source'])
    if index == 13:
        src = src.replace('hf_token = userdata.get("HF_TOKEN")', 'hf_token = None')
    if prefix:
        src = src.split(prefix, 1)[0]
    exec(compile(src, f'{SOURCE.name}:cell{index}', 'exec'), globals())

cell(3)  # imports, dispositivo, semente, configuração original
CFG.update(RUN_NAME=RUN_NAME, EPOCHS=12, PATIENCE=3,
           LR_VIT=5e-6, LR_FFT=1e-5, LR_CNN=1e-5, LR_HEAD=3e-5,
           LABEL_SMOOTHING=0.05, BATCH_SIZE=32)
CFG['START_CKPT'] = str(START)
CFG['DATA_MANIFEST'] = str(MANIFEST_PATH)
CFG['DATA_DIR'] = str(DATA_DIR)
CFG['SOURCE_SHA256'] = hashlib.sha256(SOURCE.read_bytes()).hexdigest()

if args.mode == 'prepare':
    if MANIFEST_PATH.exists():
        raise FileExistsError(f'Manifesto existente preservado: {MANIFEST_PATH}')
    cell(11)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cell(13)
    print('Dados preparados:', MANIFEST_PATH)
    raise SystemExit(0)

if not MANIFEST_PATH.is_file():
    raise FileNotFoundError(f'Prepare os dados primeiro: {MANIFEST_PATH}')

# Apenas a parte funcional do notebook: não executar células exploratórias
# que leem imagens externas, o teste ou gravam resultados antigos.
cell(11)
cell(45)
# A versão 3.3 recomprimia 100% das amostras. Metade preserva os artefatos
# originais e metade aprende tolerância a JPEG, sem mudar a avaliação.
train_tf = transforms.Compose([
    transforms.RandomCrop(CFG['CROP'], pad_if_needed=True),
    transforms.RandomHorizontalFlip(), RandomBlur(0.1), RandomDownUp(0.15),
    RandomJPEG(0.5, (55, 95)), _to_norm,
])
cell(47, 'train_pool =')
cell(49)
cell(51)

manifest = json.loads(MANIFEST_PATH.read_text(encoding='utf-8'))
train_pool = manifest['train'] + replay_rows(CFG['REPLAY_PER_CLASS'])
test_rows = manifest['test']
holdout = set(CFG['HOLDOUT'])
assert holdout == {'gpt-image-2', 'nano-banana-pro'}
assert not any(r['label'] == 1 and r['group'] in holdout for r in train_pool)
assert holdout.issubset({r['group'] for r in test_rows if r['label'] == 1})
assert all(Path(r['path']).is_file() for r in train_pool + test_rows)

tr_rows, ves_rows, vcal_rows = split_by_group(train_pool, CFG['VAL_FRAC'], CFG['SEED'])
assert tr_rows and ves_rows and vcal_rows and test_rows

# Verifica conteúdo, não apenas caminhos: cópias com nomes distintos também vazam.
def digests(rows):
    result = set()
    for row in rows:
        with open(row['path'], 'rb') as f:
            result.add(hashlib.file_digest(f, 'sha256').digest())
    return result

groups = [digests(rows) for rows in (tr_rows, ves_rows, vcal_rows, test_rows)]
for i in range(4):
    for j in range(i + 1, 4):
        assert not groups[i].intersection(groups[j]), f'Conteúdo duplicado entre partições {i} e {j}'

# Compara hashes com os conjuntos externos sem abrir imagens para análise.
external_dirs = [ROOT / 'minhas_imagens', ROOT / 'teste_externo', ROOT / 'validacao_fase3/teste_externo']
external_files = [p for folder in external_dirs if folder.exists()
                  for p in folder.rglob('*') if p.suffix.lower() in ('.jpg', '.jpeg', '.png')]
external_hashes = digests([{'path': str(p)} for p in external_files])
assert not external_hashes.intersection(groups[0] | groups[1] | groups[2]), 'Imagem externa no treino/validação'

from torch.utils.data import DataLoader, WeightedRandomSampler
nw = CFG['NUM_WORKERS']
train_loader = DataLoader(PathDataset(tr_rows, train_tf), batch_size=CFG['BATCH_SIZE'],
    sampler=WeightedRandomSampler(sampler_weights(tr_rows), len(tr_rows), replacement=True),
    num_workers=nw, pin_memory=True, drop_last=True, persistent_workers=nw > 0)
ves_loader = DataLoader(PathDataset(ves_rows, eval_tf), batch_size=64, shuffle=False, num_workers=nw)
vcal_loader = DataLoader(PathDataset(vcal_rows, multicrop_tf), batch_size=8, shuffle=False, num_workers=nw)
test_loader = DataLoader(PathDataset(test_rows, multicrop_tf), batch_size=8, shuffle=False, num_workers=nw)

# O notebook monta a arquitetura a partir do checkpoint da Fase 2; aqui trocamos
# somente os pesos iniciais pelos da variante 3.3, mantendo a mesma arquitetura.
start = torch.load(START, map_location='cpu', weights_only=False)
model.load_state_dict(start['model_state_dict'], strict=True)
del start
if args.mode == 'smoke':
    # Uma imagem de treino; sem gradientes, avaliação ou alteração dos pesos.
    x, y = next(iter(DataLoader(PathDataset(tr_rows[:1], eval_tf), batch_size=1)))
    model.eval()
    with torch.no_grad():
        logits = model(x.to(device)).float().cpu()
    assert logits.shape == (1, 2) and torch.isfinite(logits).all()
    smoke_dir = ROOT / 'modelos' / (RUN_NAME + '_smoke')
    smoke_dir.mkdir(parents=True, exist_ok=True)
    smoke_path = smoke_dir / 'smoke.json'
    if smoke_path.exists():
        raise FileExistsError(f'Smoke anterior preservado: {smoke_path}')
    with open(smoke_path, 'w', encoding='utf-8') as f:
        json.dump({'checkpoint': str(START), 'sample_label': int(y[0]),
                   'logits_shape': list(logits.shape), 'forward_finite': True}, f, indent=2)
    print('Smoke OK:', smoke_path)
    raise SystemExit(0)

if args.mode == 'train':
    RUN_DIR.mkdir(parents=True)
for dirname in ('checkpoints', 'logs', 'historicos', 'resultados'):
    (RUN_DIR / dirname).mkdir(parents=True, exist_ok=True)
LAST_CKPT = str(RUN_DIR / 'last.pt')
BEST_CKPT = str(RUN_DIR / 'best.pt')
HISTORY_CSV = str(RUN_DIR / 'history.csv')
config_path = RUN_DIR / 'experiment_config.json'
if args.mode == 'resume':
    previous = json.loads(config_path.read_text(encoding='utf-8'))
    assert previous == json.loads(json.dumps(CFG)), 'Configuração mudou desde o início; retomada recusada.'
else:
    with open(config_path, 'x', encoding='utf-8') as f:
        json.dump(CFG, f, indent=2)

train_source = ''.join(nb['cells'][55]['source'])
assert train_source.startswith('RESUME = False')
train_source = train_source.replace('RESUME = False', f'RESUME = {args.mode == "resume"}', 1)
exec(compile(train_source, f'{SOURCE.name}:cell55', 'exec'), globals())

# Calibração somente na validação B. O teste OpenFake é lido uma vez ao fim.
lg_cal, y_cal_t = collect_tta(model, vcal_loader)
T = fit_temperature(lg_cal, y_cal_t)
p_cal = F.softmax(lg_cal / T, 1)[:, 1].numpy()
thr, bal_cal = pick_threshold(p_cal, y_cal_t.numpy())
lg_te, y_te_t = collect_tta(model, test_loader)
y_te = y_te_t.numpy()
p_te = F.softmax(lg_te / T, 1)[:, 1].numpy()
auc = float(roc_auc_score(y_te, p_te))
metrics = binary_metrics(y_te, (p_te >= thr).astype(int))
metrics['auc'] = auc
metrics['threshold'] = thr
metrics['temperature'] = T
metrics['validation_calibration_bal_acc'] = bal_cal
metrics['best_epoch'] = best['epoch']
metrics['start_checkpoint'] = str(START)
with open(RUN_DIR / 'per_group.csv', 'w', newline='', encoding='utf-8') as f:
    tab = per_group_table(test_rows, p_te, thr)
    writer = csv.DictWriter(f, fieldnames=tab[0].keys())
    writer.writeheader(); writer.writerows(tab)
with open(RUN_DIR / 'test_predictions.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['path', 'label', 'group', 'prob_fake', 'pred_thr'])
    for row, prob in zip(test_rows, p_te):
        writer.writerow([row['path'], row['label'], row['group'], float(prob), int(prob >= thr)])
torch.save({'model_state_dict': model.state_dict(), 'temperature': T,
            'threshold': thr, 'cfg': cfg2, 'cfg_phase3': CFG,
            'preproc': 'lado curto<=512, JPEG q92, 5 recortes 224; logits médios'},
           RUN_DIR / 'phase3_final.pt')
with open(RUN_DIR / 'final_metrics.json.tmp', 'w', encoding='utf-8') as f:
    json.dump(metrics, f, indent=2)
os.replace(RUN_DIR / 'final_metrics.json.tmp', RUN_DIR / 'final_metrics.json')
print(json.dumps(metrics, indent=2))
print('Resultados:', RUN_DIR)
