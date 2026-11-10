"""Arquiva as saídas já executadas de um notebook, sem repetir células.

O snapshot do notebook deve conter as saídas que serão arquivadas. O log
inclui também os relatórios JSON indicados, para registrar a revisão posterior.
Não inclui código das células nem escreve em datasets ou checkpoints.
"""

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path


SECRET_PATTERNS = (
    re.compile(r"\b(?:github_pat_[A-Za-z0-9_]{8,}|ghp_[A-Za-z0-9]{8,}|hf_[A-Za-z0-9]{20,})\b"),
    re.compile(r"\bBearer\s+\S+", re.IGNORECASE),
)


def text_output(output):
    """Extrai texto apresentado ao usuário; ignora dados binários de figuras."""
    if output.get("output_type") == "stream":
        return "".join(output.get("text", []))
    if output.get("output_type") in {"execute_result", "display_data"}:
        return "".join(output.get("data", {}).get("text/plain", []))
    if output.get("output_type") == "error":
        return "\n".join(output.get("traceback", []))
    return ""


def build_log(notebook_path, reports):
    raw = notebook_path.read_bytes()
    notebook = json.loads(raw)
    lines = [
        "LOG DE EXECUÇÃO — FASE 3.9",
        f"Notebook: {notebook_path.name}",
        f"SHA-256 do notebook com saídas: {hashlib.sha256(raw).hexdigest()}",
        "Fonte: saídas persistidas no notebook versionado; células não reexecutadas.",
        "Os JSON anexados podem refletir a correção V2 posterior às saídas históricas do notebook.",
        "",
    ]
    captured = 0
    for index, cell in enumerate(notebook["cells"], 1):
        if cell.get("cell_type") != "code":
            continue
        output = "".join(text_output(item) for item in cell.get("outputs", []))
        if not output.strip():
            continue
        captured += 1
        lines.extend((f"===== CÉLULA DE CÓDIGO {captured} (posição {index}) =====", output.rstrip(), ""))
    if captured < 12:
        raise ValueError(f"Snapshot contém somente {captured} células com saídas; esperado: pelo menos 12")
    for label, path in reports.items():
        if not path.is_file():
            raise FileNotFoundError(f"Relatório {label} não encontrado: {path}")
        content = path.read_text(encoding="utf-8")
        json.loads(content)  # Garante que o bloco anexado é um relatório íntegro.
        lines.extend((f"===== RELATÓRIO {label}: {path.name} =====", content.rstrip(), ""))
    result = "\n".join(lines) + "\n"
    if any(pattern.search(result) for pattern in SECRET_PATTERNS):
        raise ValueError("Possível credencial detectada nas saídas; log não foi criado")
    return result, captured


def export(notebook_path, root, reports):
    content, captured = build_log(notebook_path, reports)
    target_dir = root / "resultados/validacao_perguntas_3_9_v1/logs"
    target_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f_UTC")
    target = target_dir / f"fase3_9_final_{stamp}.log"
    with target.open("x", encoding="utf-8") as handle:
        handle.write(content)
    return target, captured


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--v2", type=Path, required=True)
    parser.add_argument("--v3", type=Path, required=True)
    parser.add_argument("--closure", type=Path, required=True)
    args = parser.parse_args()
    target, count = export(args.notebook, args.root,
                           {"V2": args.v2, "V3": args.v3, "fechamento": args.closure})
    print(f"Log criado no Drive: {target} | células com saídas: {count}")


if __name__ == "__main__":
    main()
