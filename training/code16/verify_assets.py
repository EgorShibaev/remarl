"""Read-only verification of the reused node-local frozen backbone/runtime."""
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent


def main():
    local = Path((BASE / 'runtime/local_root.txt').read_text().strip())
    manifest = json.loads((BASE / 'runtime/checkpoint_manifest.json').read_text())
    # Keep the pinned manifest structure explicit; do not accept an unverified
    # alternative model directory if the node-local runtime has disappeared.
    entries = manifest['files']
    assert len(entries) == 14
    for entry in entries:
        name, expected = entry['local_path'], entry['sha256']
        assert name.startswith('models/LLaDA-8B-Instruct/') and '..' not in Path(name).parts
        path = local / name
        assert path.stat().st_size == entry['bytes']
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(16 * 1024 * 1024), b''):
                digest.update(block)
        assert digest.hexdigest() == expected, str(path)
    print('PINNED_BACKBONE_ASSETS_PASS', len(entries), flush=True)


if __name__ == '__main__':
    main()
