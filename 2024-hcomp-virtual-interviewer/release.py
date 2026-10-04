"""Create a release archive from an explicit reviewed file allowlist."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent
FILES = ['README.md', 'analysis.py', 'release.py', 'requirements.txt', 'CITATION.cff',
         'LICENSE-CODE', 'LICENSE-DATA', 'deposit-metadata.json', '.gitignore',
         'tests/test_end_to_end.py', 'data/descriptives.csv', 'data/kruskal.csv',
         'data/mediation.csv', 'data/posthoc.csv', 'data/item_dictionary.csv',
         'data/analysis_dictionary.csv', 'data/table_dictionary.csv', 'data/provenance.json']
FILES += [f'figures/{profile}_{kind}.{ext}' for profile in ['historical', 'corrected']
          for kind in ['means', 'indirect'] for ext in ['png', 'svg']]

def package(destination):
    destination = Path(destination).resolve()
    if destination in [(ROOT / name).resolve() for name in FILES]:
        raise ValueError('Archive cannot overwrite a release input')
    manifest = {}
    payload = {}
    for name in FILES:
        path = ROOT / name
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(ROOT):
            raise ValueError(f'Missing or unsafe release input: {name}')
        payload[name] = path.read_bytes()
        manifest[name] = hashlib.sha256(payload[name]).hexdigest()
    payload['SHA256SUMS.json'] = (json.dumps(manifest, indent=2) + '\n').encode()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in payload.items():
            info = zipfile.ZipInfo(name, date_time=(2026, 10, 4, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, content)
    return manifest

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    package(parser.parse_args().output)
