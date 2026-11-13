"""Copia notebooks e scripts de um snapshot GitHub para o Drive.

O snapshot extraído em /content é a fonte desta sincronização. Os caminhos
relativos são mantidos em TCC_Final no Drive. Se já houver um arquivo diferente
no destino, seu conteúdo anterior é preservado em codigo_historico antes da
atualização. Dados, checkpoints, resultados e credenciais não são tocados.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile


EXTENSIONS = {'.ipynb', '.py'}
FORBIDDEN_TOP = {'credenciais', 'modelos', 'resultados', 'dados_externos',
                 'checkpoints', 'codigo_sync', 'codigo_historico',
                 'dataset', 'datasets', 'data'}
IGNORED_PARTS = {'.git', '.ipynb_checkpoints', '__pycache__', '.venv', 'venv'}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_copy(source: Path, destination: Path, expected_hash: str) -> None:
    """Copia, confere e só então troca um arquivo individual no destino."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix='.tcc_sync_', dir=destination.parent,
                                     delete=False) as temporary:
        temporary_path = Path(temporary.name)
        try:
            with source.open('rb') as input_file:
                shutil.copyfileobj(input_file, temporary)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise
    try:
        if sha256(temporary_path) != expected_hash:
            raise IOError(f'Falha de integridade ao copiar {source.name}')
        os.replace(temporary_path, destination)
        if sha256(destination) != expected_hash:
            raise IOError(f'Falha de integridade no Drive: {destination}')
    finally:
        temporary_path.unlink(missing_ok=True)


def collect_files(source_root: Path) -> list[tuple[Path, Path, str, int]]:
    """Lista todos os .py e .ipynb do snapshot sem seguir links externos."""
    files = []
    for source in source_root.rglob('*'):
        if source.suffix.lower() not in EXTENSIONS:
            continue
        relative = source.relative_to(source_root)
        if any(part in IGNORED_PARTS for part in relative.parts):
            continue
        if relative.parts[0] in FORBIDDEN_TOP or relative.parts[0].startswith('dados_'):
            raise ValueError(f'Arquivo de código em pasta reservada: {relative}')
        if source.is_symlink() or not source.is_file():
            raise ValueError(f'Arquivo de código não regular: {relative}')
        files.append((source, relative, sha256(source), source.stat().st_size))
    if not files or not any(relative.suffix == '.ipynb' for _, relative, _, _ in files) \
            or not any(relative.suffix == '.py' for _, relative, _, _ in files):
        raise ValueError('Snapshot sem notebooks e scripts Python; sincronização interrompida')
    return sorted(files, key=lambda item: item[1].as_posix())


def mirror(source_root: Path, drive_root: Path, commit: str) -> dict:
    """Atualiza o espelho e grava manifesto somente depois de conferir tudo."""
    if not re.fullmatch(r'[0-9a-f]{40}', commit):
        raise ValueError('Commit deve ser um SHA-1 Git completo')
    source_root = source_root.resolve(strict=True)
    drive_root = drive_root.resolve(strict=True)
    if source_root == drive_root or source_root in drive_root.parents \
            or drive_root in source_root.parents:
        raise ValueError('Origem e Drive devem estar em árvores separadas')
    files = collect_files(source_root)
    manifest = {'branch': 'main', 'commit': commit,
                'file_count': len(files), 'total_bytes': sum(item[3] for item in files),
                'files': [{'path': relative.as_posix(), 'sha256': expected_hash,
                           'bytes': size} for _, relative, expected_hash, size in files]}
    manifest_path = drive_root / 'codigo_sync' / f'{commit}.json'
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2) + '\n').encode('utf-8')
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    if manifest_path.exists() and sha256(manifest_path) != manifest_hash:
        raise IOError(f'Manifesto existente divergente: {manifest_path}')

    summary = {'commit': commit, 'copied': 0, 'unchanged': 0, 'backed_up': 0}
    for index, (source, relative, expected_hash, size) in enumerate(files, 1):
        destination = drive_root / relative
        ancestor = drive_root
        for part in relative.parts[:-1]:
            ancestor = ancestor / part
            if ancestor.is_symlink():
                raise ValueError(f'Pasta de destino é link simbólico: {ancestor}')
        if destination.is_symlink() or (destination.exists() and not destination.is_file()):
            raise ValueError(f'Destino de código não é arquivo regular: {destination}')
        if destination.is_file():
            previous_hash = sha256(destination)
            if previous_hash == expected_hash:
                summary['unchanged'] += 1
            else:
                backup = drive_root / 'codigo_historico' / previous_hash / relative
                if backup.is_symlink():
                    raise ValueError(f'Backup é link simbólico: {backup}')
                if backup.is_file():
                    if sha256(backup) != previous_hash:
                        raise IOError(f'Backup existente com hash divergente: {backup}')
                else:
                    atomic_copy(destination, backup, previous_hash)
                summary['backed_up'] += 1
                atomic_copy(source, destination, expected_hash)
                summary['copied'] += 1
        else:
            atomic_copy(source, destination, expected_hash)
            summary['copied'] += 1
        if index % 10 == 0 or index == len(files):
            print(f'Código no Drive: {index}/{len(files)} verificados', flush=True)

    if not manifest_path.exists():
        with tempfile.NamedTemporaryFile(prefix='.tcc_manifest_',
                                         dir=manifest_path.parent if manifest_path.parent.exists() else drive_root,
                                         delete=False) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(manifest_bytes)
        try:
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            os.replace(temporary_path, manifest_path)
        finally:
            temporary_path.unlink(missing_ok=True)
    summary['file_count'] = len(files)
    summary['total_bytes'] = manifest['total_bytes']
    summary['manifest'] = str(manifest_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True,
                        help='Raiz do snapshot GitHub já extraído em /content')
    parser.add_argument('--drive-root', type=Path, required=True,
                        help='Pasta TCC_Final do Drive já montado')
    parser.add_argument('--commit', required=True, help='SHA-1 completo do snapshot')
    args = parser.parse_args()
    print(json.dumps(mirror(args.source, args.drive_root, args.commit),
                     ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
