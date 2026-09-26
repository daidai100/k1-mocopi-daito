#!/usr/bin/env python3
"""Export matched zero-update diagnostic actors from the retained initializer."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys


def prepare_checkpoints(inherited, robot, settings, source_revision, initializer_sha256):
    import torch
    from k1_motion.actuators import actuator_contract
    from k1_motion.learning import ActorCritic
    from k1_motion.model_transfer import checkpoint_hidden_sizes, initialize_model, input_mapping
    from k1_motion.observations import observation_contract

    history = inherited['observation']['history']
    observation = observation_contract(history,'preview',preview_horizon_s=.3)
    actor_size = observation['size']
    critic_size = inherited['critic_size']+actor_size-inherited['observation']['size']
    widths = checkpoint_hidden_sizes(inherited)
    model = ActorCritic(actor_size,critic_size,widths)
    transfer = initialize_model(model,inherited,observation)
    source_model = ActorCritic(inherited['actor_size'],inherited['critic_size'],widths)
    source_model.load_state_dict(inherited['model'],strict=True)
    columns = input_mapping(inherited['actor_size'],actor_size,inherited['observation'],observation,'cpu')
    added = torch.ones(actor_size,dtype=torch.bool)
    added[columns] = False
    assert torch.count_nonzero(model.actor.network[0].weight[:,added]) == 0
    generator = torch.Generator().manual_seed(713)
    samples = torch.randn((64,actor_size),generator=generator)
    perturbed = samples.clone()
    perturbed[:,added] = torch.randn((64,int(added.sum())),generator=generator)*100
    with torch.inference_mode():
        expected = source_model.actor(samples[:,columns]).tanh()
        actual = model.actor(samples).tanh()
        future_changed = model.actor(perturbed).tanh()
    inherited_error = float((expected-actual).abs().max())
    future_error = float((actual-future_changed).abs().max())
    if not torch.allclose(expected,actual,atol=2e-6,rtol=1e-6) or future_error != 0:
        raise ValueError('Expanded zero-update baseline changed inherited actions')
    provenance = {key:inherited.get(key) for key in
                  ('iteration','transitions','optimizer_steps','source_revision','backend',
                   'reference_fingerprint','model_signature','action_settings','physics_contract')}
    provenance['checkpoint_sha256'] = initializer_sha256
    provenance['training_provenance_preserved'] = True
    checkpoints = {}
    for horizon in (0.,.3):
        checkpoints[horizon] = dict(model=copy.deepcopy(model.state_dict()),stage='student',
            backend='initialization_only_no_optimizer_updates',iteration=0,transitions=0,optimizer_steps=0,
            actor_size=actor_size,critic_size=critic_size,hidden_sizes=list(widths),
            model_signature=robot.signature,
            observation=observation_contract(history,'preview',preview_horizon_s=horizon),
            reference_fingerprint=inherited['reference_fingerprint'],
            train_parents=list(inherited['train_parents']),source_revision=source_revision,
            action_settings=copy.deepcopy(settings),
            physics_contract={'backend':'matched_initializer_diagnostic',
                              'actuator':actuator_contract(robot,settings)},
            initialization_transfer=transfer,initializer_provenance=copy.deepcopy(provenance),
            optimizer_resume_supported=False,diagnostic_only=True,
            diagnostic_scope='Inherited policy under new observation/actuator contract; no new training',
            behaviorally_accepted=False,hardware_verified=False)
    proof = dict(old_actor_size=inherited['actor_size'],new_actor_size=actor_size,
                 old_critic_size=inherited['critic_size'],new_critic_size=critic_size,
                 hidden_sizes=list(widths),zero_added_actor_columns=int(added.sum()),
                 inherited_actor_action_max_error=inherited_error,
                 future_feature_action_max_error=future_error,
                 horizons_have_identical_model_tensors=True,transfer=transfer,
                 inherited_exposure={key:inherited.get(key) for key in
                                     ('iteration','transitions','optimizer_steps')},
                 new_exposure={'iterations':0,'transitions':0,'optimizer_steps':0},
                 optimizer_resume_supported=False,train_parent_count=len(inherited['train_parents']))
    return checkpoints,proof


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    bundle,output=args.bundle.resolve(),args.output.resolve()
    if output.exists():
        raise ValueError('Use a new baseline output directory')
    from freeze_source import freeze_source
    frozen,revision=freeze_source(bundle,bundle/'src/k1_motion')
    manifest=json.loads((bundle/'bundle.json').read_text())
    if revision!=manifest['source_revision']:
        raise ValueError('Bundle source revision differs')
    os.environ['K1_MOTION_ROOT']=str(bundle)
    os.environ['K1_FROZEN_SOURCE']=str(frozen/'k1_motion')
    sys.path.insert(0,str(frozen))
    import torch
    from k1_motion.actuation import action_settings
    from k1_motion.export import export_checkpoint
    from k1_motion.robot import K1Model
    torch.set_num_threads(1)
    torch.manual_seed(42)
    initializer=bundle/'initialize.pt'
    initializer_sha=hashlib.sha256(initializer.read_bytes()).hexdigest()
    if initializer_sha!=manifest['initialize_sha256']:
        raise ValueError('Initializer digest differs')
    inherited=torch.load(initializer,map_location='cpu',weights_only=True)
    if inherited.get('hidden_sizes')!=[512,256]:
        raise ValueError('Expected the retained512/256 initializer')
    robot=K1Model()
    settings=action_settings(robot,json.loads((bundle/'configs/controller-pv-official80-v1.json').read_text()))
    checkpoints,proof=prepare_checkpoints(inherited,robot,settings,revision,initializer_sha)
    output.mkdir(parents=True)
    proof.update(source_revision=revision,bundle=str(bundle),initializer=str(initializer),
                 initializer_sha256=initializer_sha,actors={})
    for horizon,checkpoint in checkpoints.items():
        directory=output/f'horizon_{round(horizon*1000):03d}'
        directory.mkdir()
        path=directory/'initializer-diagnostic.pt'
        torch.save(checkpoint,path)
        report=export_checkpoint(path,directory/'actor.pt')
        metadata_path=directory/'actor.json'
        metadata=json.loads(metadata_path.read_text())
        metadata.update(source_revision=revision,initializer_provenance=checkpoint['initializer_provenance'],
                        diagnostic_only=True,optimizer_resume_supported=False,
                        diagnostic_scope=checkpoint['diagnostic_scope'])
        metadata_path.write_text(json.dumps(metadata,indent=2)+'\n')
        proof['actors'][str(horizon)]=report
    (output/'preparation.json').write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps(proof,indent=2))


if __name__=='__main__':
    main()
