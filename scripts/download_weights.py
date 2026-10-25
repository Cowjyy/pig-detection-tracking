"""Download explicitly selected official model weights and record their SHA-256."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.runtime import ROOT,load_json,path,save_json,verify_source
from tools.preparation import sha256


def detector_urls():
    import yaml
    verify_source('mmdetection')
    desired=load_json('configs/detection/experiments.json')['models']
    urls={}
    for name,spec in desired.items():
        for meta in (ROOT/'tools/vendor/mmdetection/configs'/Path(spec['config']).parts[1]).glob('*.yml'):
            document=yaml.safe_load(meta.read_text(encoding='utf-8'))
            for record in document.get('Models',[]):
                if record.get('Config')==spec['config']:
                    urls[name]=record['Weights']
    if set(urls)!=set(desired):
        raise ValueError(f'Missing upstream weight metadata: {set(desired)-set(urls)}')
    return urls


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models',nargs='+',help='Explicit model names or all')
    parser.add_argument('--reid',action='store_true',help='MSMT17 pedestrian OSNet x0.25, not pig-trained')
    parser.add_argument('--list',action='store_true')
    args=parser.parse_args()
    urls=detector_urls()
    if args.list:
        print(json.dumps(urls,indent=2))
        return
    tasks=[]
    for model in (list(urls) if args.models==['all'] else args.models or []):
        if model not in urls:
            parser.error(f'Unknown model {model}')
        tasks.append((path(f'checkpoints/pretrained/{model}.pth'),urls[model],'COCO detector',None,[]))
    if args.reid:
        verify_source('boxmot')
        protocol=load_json('configs/tracking/experiments.json')
        source=protocol['reid_source']
        tasks.append((path(protocol['reid_weights']),source['url'],protocol['reid_domain'],
                      source['sha256'],source.get('mirrors',[])))
    if not tasks:
        parser.error('Select --models, --reid or --list')
    for target,url,domain,expected_sha256,mirrors in tasks:
        marker=target.with_suffix(target.suffix+'.json')
        if target.exists():
            digest=sha256(target)
            if marker.exists() and load_json(str(marker))['sha256']==digest and (expected_sha256 is None or digest==expected_sha256):
                print(f'Already verified: {target.name}')
                continue
            raise ValueError(f'Refusing to overwrite unverified weight: {target}')
        target.parent.mkdir(parents=True,exist_ok=True)
        print(f'Downloading {target.name}',flush=True)
        downloaded_from=None
        for attempt,candidate in enumerate([url]+mirrors,1):
            temporary=target.with_suffix(target.suffix+f'.attempt{attempt}.part')
            if temporary.exists():
                raise ValueError(f'Incomplete prior download exists: {temporary}')
            try:
                if expected_sha256 and shutil.which('curl'):
                    subprocess.run(['curl','--ipv4','--fail','--location','--silent','--show-error',
                        '--connect-timeout','10','--max-time','120','--output',str(temporary),candidate],
                        check=True,timeout=130)
                else:
                    with urllib.request.urlopen(candidate,timeout=20 if mirrors else 120) as source,temporary.open('xb') as sink:
                        while chunk:=source.read(1024*1024):
                            sink.write(chunk)
            except (OSError,TimeoutError,subprocess.SubprocessError) as error:
                print(f'Download attempt {attempt} failed: {type(error).__name__}: {error}',flush=True)
                continue
            digest=sha256(temporary)
            if expected_sha256 and digest!=expected_sha256:
                raise ValueError('Weight SHA-256 differs from the author-published file; refuse to load')
            downloaded_from=candidate
            break
        if downloaded_from is None:
            raise RuntimeError('All weight download sources failed; partial downloads are retained for inspection')
        if temporary.stat().st_size<100000:
            raise ValueError('Downloaded file is suspiciously small; do not load it as a model')
        temporary.replace(target)
        save_json(marker,dict(url=url,downloaded_from=downloaded_from,sha256=sha256(target),
            expected_sha256=expected_sha256,domain=domain,bytes=target.stat().st_size))


if __name__=='__main__':
    main()
