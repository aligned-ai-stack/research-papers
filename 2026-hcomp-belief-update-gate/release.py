"""Build only the explicitly reviewed public files. Never includes review/."""
from pathlib import Path
import argparse,hashlib,json,zipfile

ROOT=Path(__file__).resolve().parent
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'dist/belief-update-gate-public-0.1.0.zip')
    args=parser.parse_args()
    manifest=json.loads((ROOT/'PUBLIC_FILES.json').read_text())
    files=[]
    for item in manifest:
        rel=Path(item['path']);p=ROOT/rel
        if rel.is_absolute() or '..' in rel.parts or set(rel.parts)&{'review','local','dist','.git','.privacy','__pycache__'}:
            raise ValueError(f'Forbidden release path: {rel}')
        if p.is_symlink() or not p.is_file():raise ValueError(f'Missing or linked file: {rel}')
        if hashlib.sha256(p.read_bytes()).hexdigest()!=item['sha256']:
            raise ValueError(f'Changed since review: {rel}. Review the change before updating PUBLIC_FILES.json.')
        files.append((p,rel.as_posix()))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(args.output,'x',zipfile.ZIP_DEFLATED) as z:
        for p,rel in files:z.write(p,rel)
        z.write(ROOT/'PUBLIC_FILES.json','PUBLIC_FILES.json')
    with zipfile.ZipFile(args.output) as z:
        assert z.testzip() is None
        assert set(z.namelist())=={rel for _,rel in files}|{'PUBLIC_FILES.json'}
    digest=hashlib.sha256(args.output.read_bytes()).hexdigest()
    args.output.with_suffix('.zip.sha256').write_text(f'{digest}  {args.output.name}\n')
    print(f'{len(files)+1} public files; SHA-256 {digest}')
if __name__=='__main__':main()
