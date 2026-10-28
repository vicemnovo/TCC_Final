"""Registro e conferência das partições persistentes da Fase 3.4 v2."""

import json


def split_payload(train, selection, calibration, test, *, manifest_sha256,
                  sampler_policy, replay_count):
    def rows(items):
        return [{'path': str(row['path']), 'label': int(row['label']),
                 'group': str(row['group'])} for row in items]

    return {
        'manifest_sha256': manifest_sha256,
        'replay_count': replay_count,
        'sampler_policy': sampler_policy,
        'train': rows(train),
        'validation_selection': rows(selection),
        'validation_calibration': rows(calibration),
        'test_reserved': rows(test),
    }


def persist_or_verify(path, payload, *, resume=False):
    """Cria exclusivamente ou verifica o registro; nunca regrava uma partição."""
    if path.is_file():
        previous = json.loads(path.read_text(encoding='utf-8'))
        if previous != payload:
            raise ValueError(f'Partições ou dados mudaram desde o início: {path}')
        return False
    if resume:
        raise FileNotFoundError(f'Partições registradas ausentes: {path}')
    with path.open('x', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return True
