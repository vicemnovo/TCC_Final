"""Prepara e confere um benchmark externo congelado RAISE + MNW.

Este programa não treina nem calibra modelos. Só cria a versão v1 quando as
2.000 imagens e seus hashes foram verificados. Execução e downloads são manuais.
"""

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import tempfile
import time
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("/content/drive/MyDrive/TCC_Final")
NAME = "benchmark_externo_raise_mnw_2000_v1"
PER_CLASS = 1000
MNW_GROUP_COUNT = 40
MNW_PER_GROUP = PER_CLASS // MNW_GROUP_COUNT
MNW_REPO = "microsoft/MNW"
RAISE_HOSTS = {"loki.disi.unitn.it", "mmlab.science.unitn.it", "193.205.194.113"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
SEED = "tcc-externo-2000-v1-raise-mnw-2026"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stable_key(value):
    return hashlib.sha256((SEED + "|" + value).encode("utf-8")).hexdigest()


def api_json(url):
    request = urllib.request.Request(url, headers={
        "Accept": "application/vnd.github+json",
        "User-Agent": "TCC-Final-benchmark-readonly",
    })
    with urllib.request.urlopen(request, timeout=45) as response:
        return json.load(response)


def select_mnw():
    base = "https://api.github.com/repos/" + MNW_REPO
    commit = api_json(base + "/commits/main")["sha"]
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Commit MNW inválido")
    root_tree = api_json(base + "/git/trees/" + commit)
    image_dirs = [item for item in root_tree["tree"]
                  if item.get("path") == "AI_Images" and item.get("type") == "tree"]
    if len(image_dirs) != 1:
        raise ValueError("Pasta AI_Images não encontrada no commit MNW")
    image_tree = api_json(base + "/git/trees/" + image_dirs[0]["sha"] + "?recursive=1")
    if image_tree.get("truncated"):
        raise ValueError("Árvore MNW truncada; não é seguro selecionar imagens")
    by_group = defaultdict(list)
    for item in image_tree["tree"]:
        path = item.get("path", "")
        parts = path.split("/")
        if (item.get("type") == "blob" and len(parts) == 2
                and Path(parts[1]).suffix.lower() in IMAGE_EXTS):
            by_group[parts[0]].append({"name": parts[1], "path": "AI_Images/" + path})
    # Grupos com 25+ itens permitem diversidade sem amostras minúsculas.
    eligible = [group for group, files in by_group.items()
                if len(files) >= MNW_PER_GROUP and group != "Adversarial_images"]
    groups = sorted(eligible, key=stable_key)[:MNW_GROUP_COUNT]
    if len(groups) != MNW_GROUP_COUNT:
        raise ValueError(f"MNW: só {len(groups)} grupos têm {MNW_PER_GROUP}+ imagens; "
                         "não alterar seleção silenciosamente")
    rows = []
    for group in groups:
        candidates = sorted(by_group[group], key=lambda item: stable_key(item["path"]))
        for item in candidates[:MNW_PER_GROUP]:
            # O endpoint raw do GitHub resolve arquivos Git LFS para bytes da imagem.
            url = ("https://github.com/" + MNW_REPO + "/raw/" + commit + "/"
                   + urllib.parse.quote(item["path"], safe="/"))
            rows.append({"label": 1, "source": "MNW", "group": group,
                         "id": item["path"], "url": url,
                         "image": "images/fake/" + group + "/" + Path(item["name"]).stem + ".jpg"})
        print(f"MNW: {group}: {MNW_PER_GROUP} selecionadas", flush=True)
    if len(rows) != PER_CLASS or len({r["image"] for r in rows}) != PER_CLASS:
        raise ValueError("Seleção MNW incompleta ou com nomes JPEG repetidos")
    return commit, rows


def select_raise(csv_path):
    with Path(csv_path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "NEF" not in reader.fieldnames:
            raise ValueError("CSV oficial da RAISE precisa conter a coluna NEF")
        by_camera = defaultdict(list)
        for row in reader:
            url = (row.get("NEF") or "").strip()
            parsed = urllib.parse.urlparse(url)
            if parsed.scheme not in {"http", "https"} or parsed.hostname not in RAISE_HOSTS:
                continue
            if not parsed.path.lower().endswith(".nef"):
                continue
            camera = (row.get("Device") or "camera_nao_informada").strip()
            by_camera[camera].append(url)
    if sum(map(len, by_camera.values())) < PER_CLASS:
        raise ValueError(f"CSV da RAISE contém menos de {PER_CLASS} URLs NEF válidas")
    for camera in by_camera:
        by_camera[camera] = sorted(set(by_camera[camera]), key=stable_key)
    chosen = []
    cameras = sorted(by_camera)
    while len(chosen) < PER_CLASS:
        advanced = False
        for camera in cameras:
            if by_camera[camera]:
                chosen.append((camera, by_camera[camera].pop(0)))
                advanced = True
                if len(chosen) == PER_CLASS:
                    break
        if not advanced:
            raise ValueError("URLs únicas insuficientes na RAISE")
    rows = []
    for camera, url in chosen:
        filename = Path(urllib.parse.urlparse(url).path).name
        ident = Path(filename).stem
        rows.append({"label": 0, "source": "RAISE", "group": camera,
                     "id": ident, "url": url,
                     "image": "images/real/" + ident + ".jpg"})
    if len({r["id"] for r in rows}) != PER_CLASS:
        raise ValueError("RAISE contém identificadores repetidos")
    print(f"RAISE: {PER_CLASS} selecionadas | câmeras:",
          dict(Counter(r["group"] for r in rows)), flush=True)
    return rows


def save_json_exclusive(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)


def save_json_atomic(path, value):
    temp = Path(str(path) + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
    os.replace(temp, path)


def download(url, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(str(target) + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "TCC-Final-benchmark-readonly"})
    start = time.monotonic()
    last = start
    total = 0
    with urllib.request.urlopen(request, timeout=120) as response, temp.open("wb") as out:
        for block in iter(lambda: response.read(1024 * 1024), b""):
            out.write(block)
            total += len(block)
            now = time.monotonic()
            if now - last >= 30:
                print(f"  download: {target.name}: {total / 1e6:.1f} MB em {(now-start)/60:.1f} min",
                      flush=True)
                last = now
    if total == 0:
        raise ValueError(f"Arquivo vazio: {target.name}")
    # Falha fechada para páginas de erro e ponteiros Git LFS.
    if target.suffix.lower() in IMAGE_EXTS:
        from PIL import Image
        with Image.open(temp) as image:
            image.verify()
    elif target.suffix.lower() == ".nef":
        if temp.stat().st_size < 100_000:
            raise ValueError(f"RAISE retornou arquivo suspeito: {target.name}")
    os.replace(temp, target)


def verify_existing(path):
    """Recupera arquivo completo após queda entre rename e registro do hash."""
    if path.suffix.lower() in IMAGE_EXTS:
        from PIL import Image
        with Image.open(path) as image:
            image.verify()
    elif path.suffix.lower() == ".nef":
        if path.stat().st_size < 100_000:
            raise ValueError(f"RAW existente suspeito: {path}")
        import rawpy
        with rawpy.imread(str(path)) as raw:
            if raw.sizes.raw_width < 224 or raw.sizes.raw_height < 224:
                raise ValueError(f"RAW existente pequeno: {path}")


def audit_hashes(plan, hashes):
    """Diagnóstico somente-leitura dos registros; não modifica a seleção."""
    expected = {key for row in plan["rows"]
                for key in ("source:" + row["id"], row["image"])}
    missing = sorted(expected - hashes.keys())
    extra = sorted(hashes.keys() - expected)
    groups = defaultdict(list)
    for row in plan["rows"]:
        for kind, key in (("original", "source:" + row["id"]),
                          ("jpeg", row["image"])):
            if key in hashes:
                groups[hashes[key]].append({"kind": kind, "label": row["label"],
                                            "group": row["group"], "id": row["id"]})
    duplicates = [items for items in groups.values() if len(items) > 1]
    return {"expected": len(expected), "recorded": len(hashes),
            "missing": missing, "extra": extra, "duplicates": duplicates}


def repair_raise_duplicates(args):
    """Cria um plano derivado, sem mudar o plano original nem os downloads feitos."""
    work = args.root / "dados_externos" / (NAME + ".incompleto")
    original_path = work / "plan.json"
    repaired_path = work / "plan_repaired.json"
    if (args.root / "dados_externos" / NAME).exists():
        raise FileExistsError("Benchmark já congelado; não reparar após avaliação")
    if repaired_path.exists():
        print("Plano reparado existente preservado:", repaired_path, flush=True)
        return
    if not original_path.is_file() or not (work / "hashes.json").is_file():
        raise FileNotFoundError("Plano e hashes da preparação interrompida são necessários")
    if not args.raise_extra_csv.is_file():
        raise FileNotFoundError(f"CSV complementar RAISE ausente: {args.raise_extra_csv}")
    plan = json.loads(original_path.read_text(encoding="utf-8"))
    hashes = json.loads((work / "hashes.json").read_text(encoding="utf-8"))
    audit = audit_hashes(plan, hashes)
    if audit["missing"] or audit["extra"]:
        raise ValueError("Plano original com hashes faltantes/extras; não reparar automaticamente")
    rows = plan["rows"]

    def repeated_indexes(key_for_row):
        by_hash = defaultdict(list)
        for index, row in enumerate(rows):
            if row["label"] == 0:
                by_hash[hashes[key_for_row(row)]].append(index)
        return {index for indexes in by_hash.values() for index in indexes[1:]}

    retired = repeated_indexes(lambda row: "source:" + row["id"])
    if not retired or retired != repeated_indexes(lambda row: row["image"]):
        raise ValueError("Duplicatas reais não coincidem entre RAW e JPEG; investigar")
    if len(audit["duplicates"]) != 2 * len(retired):
        raise ValueError("Há outros tipos de duplicata além dos pares reais; investigar")
    original_ids = {row["id"] for row in rows}
    candidates = []
    with args.raise_extra_csv.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "NEF" not in reader.fieldnames:
            raise ValueError("CSV complementar da RAISE precisa da coluna NEF")
        for item in reader:
            url = (item.get("NEF") or "").strip()
            parsed = urllib.parse.urlparse(url)
            if (parsed.scheme not in {"http", "https"} or parsed.hostname not in RAISE_HOSTS
                    or not parsed.path.lower().endswith(".nef")):
                continue
            ident = Path(parsed.path).stem
            if ident in original_ids:
                continue
            candidates.append({"label": 0, "source": "RAISE",
                               "group": (item.get("Device") or "camera_nao_informada").strip(),
                               "id": ident, "url": url,
                               "image": "images/real/" + ident + ".jpg"})
    affected_cameras = {rows[index]["group"] for index in retired}
    candidates = sorted({row["id"]: row for row in candidates}.values(),
                        key=lambda row: (row["group"] not in affected_cameras,
                                         stable_key(row["id"])))
    if len(candidates) < len(retired):
        raise ValueError(f"CSV complementar tem só {len(candidates)} URLs novas; "
                         f"são necessárias {len(retired)} imagens únicas")
    active = [row for index, row in enumerate(rows) if index not in retired]
    used_hashes = {hashes[key] for row in active
                   for key in ("source:" + row["id"], row["image"])}
    selected = []
    args.local_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="raise_reparo_", dir=args.local_dir.parent) as temporary:
        temp_root = Path(temporary)
        for index, candidate in enumerate(candidates, 1):
            original = temp_root / "candidate.NEF"
            converted = temp_root / "candidate.jpg"
            download(candidate["url"], original)
            source_sha = digest(original)
            if source_sha not in used_hashes:
                convert(original, converted)
                jpeg_sha = digest(converted)
                if jpeg_sha not in used_hashes and jpeg_sha != source_sha:
                    selected.append({**candidate, "expected_source_sha256": source_sha,
                                     "expected_image_sha256": jpeg_sha})
                    used_hashes.update((source_sha, jpeg_sha))
                    print(f"Substituta RAISE verificada: {len(selected)}/{len(retired)} "
                          f"(candidata {index})", flush=True)
            original.unlink()
            if converted.exists():
                converted.unlink()
            if len(selected) == len(retired):
                break
    if len(selected) != len(retired):
        raise ValueError("CSV complementar não forneceu substitutas únicas suficientes")
    repaired = dict(plan)
    new_rows = list(rows)
    replacements = []
    for index, candidate in zip(sorted(retired), selected):
        replacements.append({"original_id": rows[index]["id"], "new_id": candidate["id"]})
        new_rows[index] = candidate
    repaired.update({"rows": new_rows, "parent_plan_sha256": digest(original_path),
                     "raise_extra_csv_sha256": digest(args.raise_extra_csv),
                     "retired_rows": [rows[index] for index in sorted(retired)],
                     "replacements": replacements})
    save_json_exclusive(repaired_path, repaired)
    print("Plano reparado salvo sem alterar o original:", repaired_path, flush=True)


def convert(original, target):
    from PIL import Image
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(str(target) + ".part")
    if original.suffix.lower() == ".nef":
        import rawpy
        with rawpy.imread(str(original)) as raw:
            rgb = raw.postprocess(output_bps=8, use_camera_wb=True)
        image = Image.fromarray(rgb)
    else:
        with Image.open(original) as source:
            image = source.convert("RGB")
    # Mesmo limite de resolução e codec para ambas as classes; só os derivados
    # ficam no Drive para caber na quota gratuita. Hashes dos originais são salvos.
    image.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
    image.save(temp, format="JPEG", quality=90, subsampling=0)
    with Image.open(temp) as check:
        check.verify()
    os.replace(temp, target)


def prepare(args):
    root = args.root
    final = root / "dados_externos" / NAME
    work = root / "dados_externos" / (NAME + ".incompleto")
    if final.exists():
        raise FileExistsError(f"Benchmark já congelado; não alterar: {final}")
    if work.exists():
        plan_path = work / "plan_repaired.json"
        if not plan_path.is_file():
            plan_path = work / "plan.json"
        if not plan_path.is_file():
            raise FileExistsError(f"Pasta incompleta sem plano; investigar: {work}")
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        print("Retomando a seleção já registrada:", plan_path, flush=True)
    else:
        if not args.raise_csv.is_file():
            raise FileNotFoundError("Baixe o CSV oficial da RAISE após preencher o formulário; "
                                    f"coloque-o em {args.raise_csv}")
        real = select_raise(args.raise_csv)
        commit, fake = select_mnw()
        plan = {"name": NAME, "seed": SEED, "mnw_commit": commit,
                "raise_csv_sha256": digest(args.raise_csv), "rows": real + fake}
        work.mkdir(parents=True, exist_ok=False)
        save_json_exclusive(work / "plan.json", plan)
        print("Plano fixado:", work / "plan.json", flush=True)
    if len(plan["rows"]) != 2 * PER_CLASS or Counter(
            r["label"] for r in plan["rows"]) != {0: PER_CLASS, 1: PER_CLASS}:
        raise ValueError("Plano não contém exatamente 1.000 reais e 1.000 sintéticas")
    hashes_path = work / "hashes.json"
    hashes = json.loads(hashes_path.read_text(encoding="utf-8")) if hashes_path.exists() else {}
    args.local_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="benchmark_2000_", dir=args.local_dir.parent) as temporary:
        temp_root = Path(temporary)
        for index, row in enumerate(plan["rows"], 1):
            processed = work / row["image"]
            source_key = "source:" + row["id"]
            image_key = row["image"]
            if processed.exists():
                if image_key in hashes and digest(processed) != hashes[image_key]:
                    raise ValueError(f"Imagem alterada após registro do hash: {processed}")
                if image_key not in hashes:
                    verify_existing(processed)
                    hashes[image_key] = digest(processed)
                    save_json_atomic(hashes_path, hashes)
            elif image_key in hashes:
                raise ValueError(f"Imagem registrada desapareceu: {processed}")
            if not processed.exists() or source_key not in hashes:
                suffix = Path(urllib.parse.urlparse(row["url"]).path).suffix
                original = temp_root / f"item_{index:04d}{suffix}"
                download(row["url"], original)
                source_hash = digest(original)
                if (row.get("expected_source_sha256")
                        and source_hash != row["expected_source_sha256"]):
                    raise ValueError(f"RAW substituto mudou desde o reparo: {row['id']}")
                if source_key in hashes and source_hash != hashes[source_key]:
                    raise ValueError(f"Fonte mudou desde o plano: {row['id']}")
                hashes[source_key] = source_hash
                save_json_atomic(hashes_path, hashes)
                if not processed.exists():
                    convert(original, processed)
                    hashes[image_key] = digest(processed)
                    if (row.get("expected_image_sha256")
                            and hashes[image_key] != row["expected_image_sha256"]):
                        raise ValueError(f"JPEG substituto mudou desde o reparo: {row['id']}")
                    save_json_atomic(hashes_path, hashes)
                original.unlink()
            if index % 25 == 0 or index == 2 * PER_CLASS:
                print(f"Preparação: {index}/{2 * PER_CLASS} imagens completas", flush=True)
    audit = audit_hashes(plan, hashes)
    allowed_extra = {key for row in plan.get("retired_rows", [])
                     for key in ("source:" + row["id"], row["image"])}
    unexpected_extra = sorted(set(audit["extra"]) - allowed_extra)
    if audit["missing"] or unexpected_extra or audit["duplicates"]:
        print(f"Auditoria de hashes: esperados={audit['expected']} "
              f"registrados={audit['recorded']} "
              f"faltantes={len(audit['missing'])} extras_inesperados={len(unexpected_extra)} "
              f"grupos duplicados={len(audit['duplicates'])}", flush=True)
        for index, group in enumerate(audit["duplicates"][:20], 1):
            print(f"  duplicata {index}: {group}", flush=True)
        raise ValueError("Benchmark incompleto ou com conteúdo duplicado; "
                         "a seleção permanece intacta para diagnóstico")
    verify_no_known_overlap(root, set(hashes.values()))
    manifest = {k: v for k, v in plan.items() if k != "rows"}
    manifest["rows"] = [{**row,
                         "original_sha256": hashes["source:" + row["id"]],
                         "image_sha256": hashes[row["image"]]} for row in plan["rows"]]
    manifest["processing"] = "RAW -> RGB camera WB; demais -> RGB; ambos até 2048 px -> JPEG qualidade 90"
    manifest["storage"] = "Somente JPEGs derivados no Drive; originais baixados temporariamente em /content"
    manifest["status"] = "congelado"
    frozen_manifest = work / "manifest.json"
    if frozen_manifest.exists():
        if json.loads(frozen_manifest.read_text(encoding="utf-8")) != manifest:
            raise ValueError(f"Manifesto incompleto difere do plano: {frozen_manifest}")
    else:
        save_json_exclusive(frozen_manifest, manifest)
    work.rename(final)
    print("Benchmark congelado:", final, flush=True)


def pilot(args):
    if not args.raise_csv.is_file():
        raise FileNotFoundError(f"CSV oficial RAISE ausente: {args.raise_csv}")
    real = select_raise(args.raise_csv)
    commit, fake = select_mnw()
    target = args.local_dir.parent / "benchmark_2000_piloto"
    target.mkdir(parents=True, exist_ok=True)
    for row in (real[0], fake[0]):
        original = target / ("original_" + row["source"] +
                             Path(urllib.parse.urlparse(row["url"]).path).suffix)
        if not original.exists():
            download(row["url"], original)
        converted = target / row["image"]
        if not converted.exists():
            convert(original, converted)
        print(f"Piloto {row['source']}: {original.stat().st_size / 1e6:.2f} MB; "
              f"JPEG legível; SHA-256={digest(original)}", flush=True)
    print("Piloto OK; commit MNW:", commit, flush=True)


def verify_no_known_overlap(root, new_hashes):
    """Bloqueia sobreposição exata com OpenFake registrado e as 20 legadas."""
    split_path = root / "modelos/fase3_4_finetune_v2_sqrt_sampler/split_manifest.json"
    manifest_path = root / "dados_fase3_4/manifest.json"
    if not split_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("Manifestos OpenFake v2 ausentes; não congelar teste sem auditoria")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if digest(manifest_path) != split["manifest_sha256"]:
        raise ValueError("Manifesto OpenFake mudou desde o treino v2")
    keys = ("train", "validation_selection", "validation_calibration", "test_reserved")
    paths = [Path(row["path"]) for key in keys for row in split[key]]
    if len(paths) != 3979:
        raise ValueError(f"Partições OpenFake inesperadas: {len(paths)}")
    for index, path in enumerate(paths, 1):
        if not path.is_file():
            raise FileNotFoundError(f"Imagem OpenFake ausente: {path}")
        if digest(path) in new_hashes:
            raise ValueError(f"Sobreposição exata com OpenFake: {path}")
        if index % 500 == 0:
            print(f"Auditoria de sobreposição: {index}/3979", flush=True)
    historical = root / "validacao_fase3/teste_externo"
    if not historical.is_dir():
        raise FileNotFoundError("20 imagens históricas ausentes; não congelar sem auditoria")
    old = [p for p in historical.iterdir() if p.suffix.lower() in IMAGE_EXTS]
    if len(old) != 20:
        raise ValueError(f"Conjunto histórico inesperado: {len(old)} imagens")
    for path in old:
        if digest(path) in new_hashes:
            raise ValueError(f"Sobreposição exata com avaliação histórica: {path}")
    print("Sem sobreposição exata com OpenFake v2 nem com as 20 imagens históricas.", flush=True)


def verify_and_stage(args):
    source = args.root / "dados_externos" / NAME
    manifest_path = source / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Benchmark de 400 ainda não congelado: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = manifest["rows"]
    if len(rows) != 2 * PER_CLASS or Counter(r["label"] for r in rows) != {
            0: PER_CLASS, 1: PER_CLASS}:
        raise ValueError("Contagem/rótulos do benchmark alterados")
    target = args.local_dir
    if source.resolve() == target.resolve():
        raise ValueError("Diretório local precisa ser separado do Drive")
    staged_manifest = target / "manifest.json"
    if staged_manifest.exists() and digest(staged_manifest) != digest(manifest_path):
        raise ValueError(f"Manifesto local de outra versão; não substituir: {staged_manifest}")
    needed = sum((source / row["image"]).stat().st_size for row in rows
                 if not (target / row["image"]).exists())
    target.parent.mkdir(parents=True, exist_ok=True)
    if needed and shutil.disk_usage(target.parent).free < needed * 1.1:
        raise OSError(f"Espaço insuficiente em {target.parent}; precisa de ~{needed / 1e9:.2f} GB")
    for index, row in enumerate(rows, 1):
        rel = Path(row["image"])
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError("Caminho inseguro no manifesto")
        src = source / rel
        if not src.is_file() or digest(src) != row["image_sha256"]:
            raise ValueError(f"Imagem do Drive ausente ou alterada: {rel}")
        dst = target / rel
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        if digest(dst) != row["image_sha256"]:
            raise ValueError(f"Imagem local diferente do manifesto: {rel}")
        if index % 50 == 0:
            print(f"Cópia/verificação: {index}/{2 * PER_CLASS}", flush=True)
    target.mkdir(parents=True, exist_ok=True)
    if not staged_manifest.exists():
        shutil.copy2(manifest_path, staged_manifest)
    print("Cópia local pronta, 1.000 reais + 1.000 sintéticas:", target, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("pilot", "prepare", "repair", "stage"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--raise-csv", type=Path, default=ROOT / "entradas" / "RAISE_1k.csv")
    parser.add_argument("--raise-extra-csv", type=Path,
                        default=ROOT / "entradas" / "RAISE_2k.csv")
    parser.add_argument("--local-dir", type=Path, default=Path("/content") / NAME)
    args = parser.parse_args()
    if not args.root.is_dir():
        raise FileNotFoundError(f"Drive não montado: {args.root}")
    if args.mode == "pilot":
        pilot(args)
    elif args.mode == "prepare":
        prepare(args)
    elif args.mode == "repair":
        repair_raise_duplicates(args)
    else:
        verify_and_stage(args)


if __name__ == "__main__":
    main()
