"""Check the shared split and result snapshots without loading raw data or weights."""
from pathlib import Path
import csv, hashlib, json, sys

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/results/run_20261022_01'
def read(path): return json.loads(path.read_text(encoding='utf-8'))
def rows(path):
    with path.open(newline='',encoding='utf-8-sig') as f: return list(csv.DictReader(f))
def require(ok,message):
    if not ok: raise ValueError(message)

def main():
    split=read(ROOT/'datasets/splits/protocol_v2/detection/splits.json')
    ids=[]
    for name,expected in [('train',660),('val',195),('test',97),('reserve',63)]:
        members=split['reserved_ids'] if name=='reserve' else split['splits'][name]
        require(len(members)==expected,f'Unexpected {name} count')
        table=rows(ROOT/f'datasets/splits/protocol_v2/detection/{name}.csv')
        require([x['image_id'] for x in table]==members,f'{name} CSV differs from JSON')
        txt=(ROOT/f'datasets/splits/protocol_v2/detection/{name}.txt').read_text().splitlines()
        require(txt==[x['image'] for x in table],f'{name} TXT differs from CSV')
        ids+=members
    require(len(set(ids))==len(ids)==1015,'Split members overlap or are missing')
    track=read(ROOT/'datasets/splits/protocol_v2/tracking/splits.json')
    require(track['tuning_sequences']==['191','194'],'Tuning membership changed')
    require(track['test_sequences']==['192','193','195','196','197'],'Test membership changed')
    require(not set(track['tuning_sequences']) & set(track['test_sequences']),'Video leakage')
    result=read(OUT/'results.json')
    require(result['seeds']==[20261022],'Seed protocol changed')
    require(len(result['detection'])==len(result['tracking'])==6,'Missing method')
    require(len(result['tracking_per_sequence'])==30,'Missing per-sequence result')
    require(len(rows(OUT/'tracking_tuning.csv'))==18,'Missing parameter trial')
    require(len(rows(OUT/'tracking_decomposition.csv'))==6,'Missing decomposition')
    for x in result['tracking']:
        frozen=read(OUT/f'frozen_parameters/{x["method"]}.json')
        require(frozen['method']==x['method'] and frozen['selected_on']=='tuning','Invalid frozen selection')
    export=read(OUT/'export_manifest.json')
    for x in export['exports']:
        p=(ROOT/x['path']).resolve()
        require(p.is_relative_to(OUT.resolve()),'Snapshot path escapes result directory')
        require(hashlib.sha256(p.read_bytes()).hexdigest()==x['sha256'],f'Snapshot hash changed: {x["path"]}')
    inventory=read(ROOT/'datasets/raw/inventory.json')
    require(len(inventory)==2051,'Unexpected raw inventory baseline')
    print(json.dumps(dict(status='passed',detection_members=1015,tracking_sequences=7,
        detectors=6,trackers=6,trials=18,per_sequence_rows=30,result_hashes=len(export['exports']))))

if __name__=='__main__': main()
