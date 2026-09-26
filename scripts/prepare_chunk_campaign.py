#!/usr/bin/env python3
"""Stage immutable long clips, shared tensors and equal-capacity chunk actors."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

from freeze_source import freeze_source
from run_padded_causal_campaign import verify_membership
from run_sustained_campaign import write

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def relocate_cache(source,original,portable,destination,robot):
    """Rehome metadata only after validating exact source cache and row identity."""
    import torch
    from k1_motion.reference_cache import load_reference_cache, cache_contract
    source,original,portable,destination = map(Path,(source,original,portable,destination))
    if destination.exists():
        raise ValueError('Refusing to overwrite a reference cache')
    old = load_reference_cache(source,original,robot,'cpu')
    rows = [json.loads(line) for line in (portable/'index.jsonl').read_text().splitlines()]
    def strip(row):
        return {k:v for k,v in row.items() if k != 'reference_path'}
    if len(rows) != len(old.rows) or any(strip(a) != strip(b) for a,b in zip(old.rows,rows)):
        raise ValueError('Portable manifest differs beyond reference paths')
    payload = torch.load(source,map_location='cpu',weights_only=True,mmap=True)
    payload['state']['rows'] = rows
    payload['state']['fingerprint'] = hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()
    payload['contract'] = cache_contract(portable,robot,False,'packed')
    temporary = destination.with_suffix('.partial')
    torch.save(payload,temporary)
    temporary.replace(destination)
    report = json.loads(source.with_suffix('.json').read_text())
    report.update(payload['contract'],fingerprint=payload['state']['fingerprint'],
        file_bytes=destination.stat().st_size,relocated_from=str(source.resolve()),
        source_proof='Exact admitted row identity/order except reference_path; original cache verified',
        tensor_transformations=0)
    write(destination.with_suffix('.json'),report)
    load_reference_cache(destination,portable,robot,'cpu')
    return report


def copy_clip(source,destination):
    destination.parent.mkdir(parents=True,exist_ok=True)
    if not destination.exists():
        try:
            os.link(source,destination)
        except OSError:
            shutil.copy2(source,destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--padded-inputs',type=Path,required=True)
    parser.add_argument('--prior-campaign',type=Path,required=True)
    parser.add_argument('--initializer',type=Path,required=True)
    parser.add_argument('--old-package',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--remote-root',type=Path,required=True)
    args = parser.parse_args()
    import torch
    from k1_motion.action_chunks import make_model, chunk_contract
    from k1_motion.model_transfer import initialize_model
    from k1_motion.reference_cache import rebind_reference_cache
    from k1_motion.robot import K1Model
    torch.set_num_threads(1)
    output = args.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    bundle = output/'bundle'
    bundle.mkdir()
    remote = args.remote_root/'bundle'
    padded = args.padded_inputs.resolve(strict=True)
    prior = args.prior_campaign.resolve(strict=True)
    rows = [json.loads(s) for s in (padded/'library/index.jsonl').read_text().splitlines()]
    all_rows = [json.loads(s) for s in (padded/'all-padded/index.jsonl').read_text().splitlines()]
    curriculum = json.loads((prior/'inputs/curriculum.json').read_text())
    verify_membership(rows,all_rows,curriculum)
    original_manifest = hashlib.sha256((padded/'library/index.jsonl').read_bytes()).hexdigest()
    audit = json.loads((padded/'report.json').read_text())
    if original_manifest != audit['training_manifest_sha256'] or audit['invalid_ticks']:
        raise ValueError('Padded data differs from its audited manifest')
    library = bundle/'library'
    library.mkdir()
    portable = []
    for row in rows:
        relative = Path('clips')/(row['id']+'.npz')
        copy_clip(Path(row['reference_path']),library/relative)
        portable.append(dict(row,reference_path=str(remote/'library'/relative)))
    (library/'index.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in portable))
    write(library/'staging.json',dict(originals=len(rows),source_manifest_sha256=original_manifest,
        portable_manifest_sha256=hashlib.sha256((library/'index.jsonl').read_bytes()).hexdigest(),
        original_durations_strictly_greater_than_s=10,tensor_transformations=0,
        payload_bytes=sum((library/'clips'/(r['id']+'.npz')).stat().st_size for r in rows)))
    inputs = bundle/'inputs'
    inputs.mkdir()
    (inputs/'library').symlink_to('../library',target_is_directory=True)
    write(inputs/'curriculum.json',curriculum)
    by_id = {r['id']:r for r in portable}
    for role in ('training','development'):
        panel = json.loads((prior/f'inputs/{role}-panel.json').read_text())
        for row in panel:
            if role == 'training':
                row['reference_path'] = by_id[row['id']]['reference_path']
            else:
                relative = Path('development-clips')/(row['id']+'.npz')
                copy_clip(Path(row['reference_path']),bundle/relative)
                row['reference_path'] = str(remote/relative)
        write(inputs/f'{role}-panel.json',panel)
    robot = K1Model()
    rebound = output/'reference-cache-runtime.pt'
    rebind_reference_cache(padded/'reference-cache.pt',rebound,args.old_package,padded/'library',robot)
    relocate_cache(rebound,padded/'library',library,output/'reference-cache.pt',robot)
    snapshot,revision = freeze_source(ROOT)
    shutil.copytree(snapshot/'k1_motion',bundle/'src/k1_motion')
    shutil.copytree(ROOT/'configs',bundle/'configs')
    assets = (ROOT/json.loads((ROOT/'configs/k1.json').read_text())['model']).parent
    shutil.copytree(assets,bundle/assets.relative_to(ROOT))
    scripts = bundle/'scripts'
    scripts.mkdir()
    for source in (ROOT/'scripts').glob('*.py'):
        shutil.copy2(source,scripts/source.name)
    initial = torch.load(args.initializer,map_location='cpu',weights_only=True)
    if initial['observation'].get('preview_horizon_s') != .3 or initial['reward_settings'].get('reference_scale'):
        raise ValueError('Expected original-scale 300 ms preview initializer')
    write(bundle/'controller.json',initial['action_settings'])
    torch.manual_seed(45)
    common = make_model(initial['actor_size'],initial['critic_size'],(2048,1024),2)
    transfer = initialize_model(common,initial,initial['observation'])
    initializers = []
    for length in (2,4,8):
        model = make_model(initial['actor_size'],initial['critic_size'],(2048,1024),length)
        model.load_state_dict(common.state_dict(),strict=True)
        checkpoint = {k:v for k,v in initial.items() if k not in ('optimizer','curriculum_state','reference_exposure')}
        checkpoint.update(model=model.state_dict(),hidden_sizes=[2048,1024],action_chunk=chunk_contract(length),
            iteration=0,transitions=0,optimizer_steps=0,source_revision=revision,
            train_parents=sorted(set(initial['train_parents']) | {r['capture_group'] for r in rows}))
        destination = bundle/f'initializers/chunk_{length}.pt'
        destination.parent.mkdir(exist_ok=True)
        torch.save(checkpoint,destination)
        initializers.append(dict(length=length,actor_parameters=sum(p.numel() for p in model.actor.parameters()),
            total_parameters=sum(p.numel() for p in model.parameters()),path=str(remote/f'initializers/chunk_{length}.pt')))
    write(bundle/'bundle.json',dict(version='all-long-action-chunks-v1',source_revision=revision,
        original_manifest_sha256=original_manifest,originals=len(rows),initializers=initializers,
        original_initializer=str(args.initializer.resolve()),
        original_initializer_sha256=hashlib.sha256(args.initializer.read_bytes()).hexdigest(),
        widening_verification=transfer,actor_parameters=initializers[0]['actor_parameters'],
        parameters_identical_across_chunks=True,behaviorally_accepted=False))
    print(json.dumps(dict(output=str(output),source_revision=revision,originals=len(rows),initializers=initializers)))


if __name__ == '__main__':
    main()
