"""Fetch fixed official source archives into tools/vendor; no installation or training."""
import argparse
import io
import json
from pathlib import Path
import sys
import tarfile
import urllib.request

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.preparation import ROOT, sha256, write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--only',nargs='+')
    args=parser.parse_args()
    sources=json.loads((ROOT/'configs/upstream_sources.json').read_text())
    for name,spec in sources.items():
        if args.only and name not in args.only:
            continue
        destination=ROOT/'tools/vendor'/name
        marker=destination/'SOURCE_LOCK.json'
        if destination.exists():
            if marker.is_file() and json.loads(marker.read_text())['commit']==spec['commit']:
                print(f'{name}: pinned source already present',flush=True)
                continue
            raise ValueError(f'Refusing to overwrite unverified source: {destination}')
        url=f'https://codeload.github.com/{spec["repo"]}/tar.gz/{spec["commit"]}'
        print(f'Downloading {name} @ {spec["commit"]}',flush=True)
        with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'pig-research-bootstrap'}),timeout=120) as response:
            data=response.read()
        destination.mkdir(parents=True)
        with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as archive:
            for member in archive.getmembers():
                parts=Path(member.name).parts[1:]
                if not parts:
                    continue
                if member.issym() or member.islnk():
                    continue
                target=(destination/Path(*parts)).resolve()
                if not target.is_relative_to(destination.resolve()):
                    raise ValueError('Unsafe archive path')
                if member.isdir():
                    target.mkdir(parents=True,exist_ok=True)
                elif member.isfile():
                    target.parent.mkdir(parents=True,exist_ok=True)
                    with archive.extractfile(member) as source,target.open('xb') as stream:
                        stream.write(source.read())
        import hashlib
        write_json(marker,dict(**spec,archive_url=url,archive_sha256=hashlib.sha256(data).hexdigest()))
        print(f'{name}: source ready',flush=True)


if __name__=='__main__':
    main()
