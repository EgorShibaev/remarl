"""Verify the released policy archive before loading it."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    args=parser.parse_args()
    manifest=json.loads((Path(__file__).parent/'checkpoints/manifest.json').read_text())
    for item in manifest['checkpoints']+[manifest['training_architecture_asset']]:
        path=args.directory/item['file']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:
            raise ValueError(f'Checksum mismatch: {path}')
        print('OK',item['file'])


if __name__=='__main__':main()
