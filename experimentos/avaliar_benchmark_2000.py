"""Reavalia checkpoints congelados das Fases 1–3.4 no MESMO benchmark de 2.000.

Resultados antigos nunca são alterados. Nenhuma imagem externa é usada para
treino, calibração, seleção de pesos ou ajuste de limiar.
"""

import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path

from benchmark_2000 import NAME, ROOT, PER_CLASS, digest

MODELS = {
    "fase1": ("phase1_best_model.pt", "resize"),
    "fase2": ("fase2_gate_vit_fft_cnn/phase2_final.pt", "resize"),
    "fase3_1": ("fase3_multigerador/phase3_final.pt", "five_crop"),
    "fase3_2": ("fase3_holdout_v2/phase3_final.pt", "five_crop"),
    "fase3_3": ("fase3_3_compressao_v1/phase3_final.pt", "five_crop"),
    "fase3_4_v1": ("fase3_4_finetune_v1/phase3_final.pt", "five_crop"),
    "fase3_4_v2": ("fase3_4_finetune_v2_sqrt_sampler/phase3_final.pt", "five_crop"),
}


def architecture_sources(code_root):
    """Extrai as classes originais de cada fase sem executar células de treino."""
    phase1_source = code_root / "Notebooks/fase_1/fase1_vit.ipynb"
    phase2_source = code_root / "Notebooks/fase_3/fase3.3_compressao.ipynb"
    if not phase1_source.is_file() or not phase2_source.is_file():
        raise FileNotFoundError("Notebook original da Fase 1 ou 3.3 ausente no snapshot")
    phase1_cells = json.loads(phase1_source.read_text(encoding="utf-8"))["cells"]
    classes = []
    for index in (20, 22, 24):
        cell = ast.parse("".join(phase1_cells[index]["source"]))
        classes.extend(node for node in cell.body if isinstance(node, ast.ClassDef))
    if "Phase1Detector" not in {node.name for node in classes}:
        raise RuntimeError("Classe Phase1Detector não encontrada no notebook da Fase 1")
    phase1_module = ast.fix_missing_locations(ast.Module(body=classes, type_ignores=[]))
    phase2_cells = json.loads(phase2_source.read_text(encoding="utf-8"))["cells"]
    phase2_cell = "".join(phase2_cells[49]["source"])
    marker = "# ---------- carrega o checkpoint final da Fase 2 ----------"
    if marker not in phase2_cell:
        raise RuntimeError("Marcador de arquitetura da célula 49 mudou")
    phase2_code = phase2_cell.split(marker, 1)[0]
    if "Phase2Detector" not in {node.name for node in ast.parse(phase2_code).body
                                if isinstance(node, ast.ClassDef)}:
        raise RuntimeError("Classe Phase2Detector não encontrada no notebook da Fase 3.3")
    architecture_sha = hashlib.sha256(
        (ast.dump(phase1_module, include_attributes=False) + phase2_code).encode("utf-8")
    ).hexdigest()
    return phase1_module, phase2_code, architecture_sha


def load_architecture(code_root, device):
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

    phase1_module, phase2_code, architecture_sha = architecture_sources(code_root)
    namespace = {"torch": torch, "nn": nn, "F": F, "np": np, "device": device,
                 "efficientnet_b0": efficientnet_b0,
                 "EfficientNet_B0_Weights": EfficientNet_B0_Weights}
    phase1_namespace = dict(namespace)
    exec(compile(phase1_module, "fase1_vit.ipynb:classes", "exec"), phase1_namespace)
    exec(compile(phase2_code, "fase3.3_compressao.ipynb:cell49", "exec"), namespace)
    return phase1_namespace["Phase1Detector"], namespace["Phase2Detector"], architecture_sha


def load_model(name, checkpoint_path, phase1_cls, phase2_cls, device):
    import torch

    # Arquivos do próprio projeto são confiáveis; não abrir checkpoints de terceiros.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if name == "fase1":
        if not isinstance(checkpoint, dict) or "model_state_dict" in checkpoint:
            raise ValueError("Checkpoint da Fase 1 não é o state_dict esperado")
        model = phase1_cls()
        model.load_state_dict(checkpoint, strict=True)
        temperature, threshold = 1.0, 0.5
        note = "Fase 1: checkpoint sem calibração persistida; usa T=1 e limiar=0,5."
    else:
        if not {"model_state_dict", "cfg", "temperature", "threshold"} <= checkpoint.keys():
            raise ValueError(f"Checkpoint incompleto: {checkpoint_path}")
        cfg = checkpoint["cfg"]
        model = phase2_cls(branches=cfg["BRANCHES"], fusion=cfg["FUSION"],
                           proj_dim=cfg["PROJ_DIM"], pretrained_cnn=False)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        temperature = float(checkpoint["temperature"])
        threshold = float(checkpoint["threshold"])
        note = "Temperatura e limiar lidos do checkpoint, sem recalibração."
    if not 0 < temperature < 10 or not 0 < threshold < 1:
        raise ValueError("Temperatura ou limiar inválido")
    return model.to(device).eval(), temperature, threshold, note


def checked_rows(local_dir):
    manifest_path = local_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Execute benchmark_2000.py stage primeiro: {manifest_path}")
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = data["rows"]
    if len(rows) != 2 * PER_CLASS or sum(r["label"] == 0 for r in rows) != PER_CLASS or sum(
            r["label"] == 1 for r in rows) != PER_CLASS:
        raise ValueError("Benchmark não contém 1.000 reais e 1.000 sintéticas")
    for index, row in enumerate(rows, 1):
        rel = Path(row["image"])
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError("Caminho inseguro no manifesto")
        path = local_dir / rel
        if not path.is_file() or digest(path) != row["image_sha256"]:
            raise ValueError(f"Imagem alterada ou ausente: {rel}")
        if index % 100 == 0:
            print(f"Integridade antes da avaliação: {index}/{2 * PER_CLASS}", flush=True)
    return manifest_path, rows


def evaluate(args):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from sklearn.metrics import roc_auc_score
    from torchvision import transforms

    checkpoint = args.root / "modelos" / MODELS[args.model][0]
    output = args.root / "resultados" / NAME / args.model
    if output.exists():
        raise FileExistsError(f"Resultado anterior preservado: {output}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint ausente: {checkpoint}")
    manifest_path, rows = checked_rows(args.local_dir)
    manifest_sha = digest(manifest_path)
    script_sha = digest(Path(__file__))
    result_root = args.root / "resultados" / NAME
    for prior in result_root.glob("*/metrics.json"):
        recorded = json.loads(prior.read_text(encoding="utf-8"))
        if recorded.get("manifest_sha256") != manifest_sha or recorded.get(
                "evaluation_script_sha256") != script_sha:
            raise ValueError(f"Protocolo/manifesto difere de avaliação anterior: {prior}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Dispositivo:", device, flush=True)
    phase1_cls, phase2_cls, arch_sha = load_architecture(args.code_root, device)
    for prior in result_root.glob("*/metrics.json"):
        recorded = json.loads(prior.read_text(encoding="utf-8"))
        if recorded.get("architecture_source_sha256") != arch_sha:
            raise ValueError(f"Código da arquitetura mudou desde: {prior}")
    model, temperature, threshold, note = load_model(
        args.model, checkpoint, phase1_cls, phase2_cls, device)
    print(f"{args.model}: checkpoint compatível carregado | T={temperature:.6f} | "
          f"limiar={threshold:.6f}", flush=True)
    mode = MODELS[args.model][1]
    norm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    probabilities = []
    labels = []
    with torch.inference_mode():
        for start in range(0, 2 * PER_CLASS, 4):
            block = rows[start:start + 4]
            tensors = []
            for row in block:
                with Image.open(args.local_dir / row["image"]) as opened:
                    image = opened.convert("RGB")
                if mode == "resize":
                    tensors.append(norm(image.resize((224, 224))))
                else:
                    if min(image.size) < 224:
                        raise ValueError(f"Imagem pequena para cinco recortes: {row['image']}")
                    tensors.extend(norm(crop) for crop in transforms.FiveCrop(224)(image))
            batch = torch.stack(tensors).to(device)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(batch).float()
            if mode == "five_crop":
                logits = logits.reshape(len(block), 5, 2).mean(1)
            probabilities.extend(F.softmax(logits / temperature, dim=1)[:, 1].cpu().tolist())
            labels.extend(row["label"] for row in block)
            if (start + len(block)) % 100 == 0:
                print(f"{args.model}: {start + len(block)}/{2 * PER_CLASS} inferências", flush=True)
    y = np.asarray(labels)
    p = np.asarray(probabilities)
    pred = (p >= threshold).astype(int)
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tp = int(((y == 1) & (pred == 1)).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / PER_CLASS
    tnr = tn / PER_CLASS
    metrics = {
        "model": args.model, "n": 2 * PER_CLASS, "real": PER_CLASS, "fake": PER_CLASS,
        "accuracy": (tn + tp) / (2 * PER_CLASS), "auc": float(roc_auc_score(y, p)),
        "balanced_accuracy": (recall + tnr) / 2,
        "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / max(1e-12, precision + recall),
        "specificity": tnr, "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "temperature_frozen": temperature, "threshold_frozen": threshold,
        "checkpoint": str(checkpoint), "checkpoint_sha256": digest(checkpoint),
        "manifest_sha256": manifest_sha,
        "evaluation_script_sha256": script_sha,
        "architecture_source_sha256": arch_sha,
        "protocol": mode, "note": note,
    }
    output.mkdir(parents=True, exist_ok=False)
    with (output / "predictions.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "id", "source", "group", "label_real", "image_sha256",
            "prob_fake", "pred", "correct",
        ])
        writer.writeheader()
        for row, prob, guess in zip(rows, p, pred):
            writer.writerow({
                "id": row["id"], "source": row["source"], "group": row["group"],
                "label_real": row["label"], "image_sha256": row["image_sha256"],
                "prob_fake": float(prob), "pred": int(guess),
                "correct": int(guess == row["label"]),
            })
    with (output / "metrics.json").open("x", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)
    print(json.dumps(metrics, indent=2, ensure_ascii=False), flush=True)
    print("Resultado novo salvo:", output, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=tuple(MODELS))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--local-dir", type=Path, default=Path("/content") / NAME)
    parser.add_argument("--code-root", type=Path, required=True)
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
