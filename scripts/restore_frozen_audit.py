"""Recover the frozen audit's byte layout across platforms, without changing values."""
from pathlib import Path
import argparse, hashlib, json, sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.preparation import ROOT,load_checked_records

def frozen_bytes(data,expected):
    lf=data.replace(b'\r\n',b'\n')
    for candidate in (data,lf,lf.replace(b'\n',b'\r\n')):
        if hashlib.sha256(candidate).hexdigest()==expected:
            return candidate
    raise ValueError('Audit differs beyond newline style; verify original data and audit implementation')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--records',default='experiments/preparation/audit_20261022_03/detection_records.json')
    p.add_argument('--output',default='.cache/frozen_detection_records.json')
    p.add_argument('--tracking-gt',action='store_true',help='Restore only newline bytes of generated GT to frozen content hashes')
    a=p.parse_args()
    if a.tracking_gt:
        protocol=json.loads((ROOT/'datasets/splits/protocol_v2/tracking/splits.json').read_text())
        manifest_path=ROOT/'datasets/annotations/mot_clean_v1/manifest.json'
        manifest=json.loads(manifest_path.read_text())
        by_id={r['sequence']:r for r in manifest['sequences']}
        changes=[]
        for record in protocol['sequences']:
            target=(ROOT/record['gt']).resolve()
            if not target.is_relative_to((ROOT/'datasets/annotations/mot_clean_v1').resolve()):
                raise ValueError('Refusing to write outside the derived GT directory')
            current=target.read_bytes()
            candidate=frozen_bytes(current,record['gt_sha256'])
            item=by_id[record['sequence']]
            if (ROOT/item['gt']).resolve()!=target: raise ValueError('GT manifests disagree about output paths')
            if current!=candidate:
                changes.append((target,candidate,item,hashlib.sha256(current).hexdigest(),record['gt_sha256']))
        # Validate every sequence before any derived output is changed.
        if changes:
            backup=manifest_path.with_name('manifest_before_newline_restore.json')
            if backup.exists(): raise ValueError('A prior byte-layout restoration exists; inspect it first')
            backup.write_bytes(manifest_path.read_bytes())
            for target,candidate,item,before,after in changes:
                target.write_bytes(candidate)
                item['gt_sha256']=after
            manifest['byte_layout_restoration']=[dict(sequence=item['sequence'],before_sha256=before,
                after_sha256=after,policy='LF/CRLF only; values unchanged') for _,_,item,before,after in changes]
            manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
        print(json.dumps(dict(sequences_verified=len(protocol['sequences']),newline_layouts_restored=len(changes))))
        return
    source=(ROOT/a.records).resolve()
    load_checked_records(source)
    split=json.loads((ROOT/'datasets/splits/protocol_v2/detection/splits.json').read_text())
    data=frozen_bytes(source.read_bytes(),split['records_sha256'])
    target=(ROOT/a.output).resolve()
    if not target.is_relative_to((ROOT/'.cache').resolve()) or target==(ROOT/'.cache').resolve():
        raise ValueError('Output must stay inside project .cache/')
    if target.exists():
        if target.read_bytes()!=data: raise ValueError('Refusing to overwrite a different existing file')
    else:
        target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as f: f.write(data)
    print(target.relative_to(ROOT).as_posix())

if __name__=='__main__': main()
