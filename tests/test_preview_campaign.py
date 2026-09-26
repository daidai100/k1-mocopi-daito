"""Risks: future masking changes width, safety/speed confounds, unsafe resume,
or a dry-run launches GPU work. Written before the preview launcher.
"""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

PATH = Path(__file__).resolve().parents[1]/'scripts/run_preview_campaign.py'


def launcher():
    spec=importlib.util.spec_from_file_location('preview_campaign_launcher',PATH)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def option(command,key):
    return command[command.index(key)+1]


def test_fixed_four_arms_keep_safety_actuation_and_ppo_common(tmp_path):
    m=launcher()
    names=['simple_masked','simple_preview','world_masked','world_preview']
    assert list(m.TREATMENTS)==names
    plans=m.build_plans(tmp_path,tmp_path/'out',tmp_path/'cache','server',names,125)
    for i,p in enumerate(plans):
        c=p['command']
        assert option(c,'--reward-profile')==('simple-track-v2' if i<2 else 'world-body-v1')
        assert float(option(c,'--preview-horizon-s'))==(0. if i%2==0 else .3)
        assert option(c,'--observation-profile')=='preview'
        assert option(c,'--safety-profile')=='casual-safe-v1'
        assert option(c,'--action-settings').endswith('controller-pv-official80-v1.json')
        assert option(c,'--self-collision-weight')=='0'
        assert option(c,'--first-collision-penalty')=='0'
        assert '--command-velocity-limit' not in c and '--corruption' not in c
        assert '--residual-scale' not in c and '--allow-env-resize' not in c
        for flag,value in {'--num-envs':'2048','--horizon':'32','--history':'10','--minibatch':'4096',
                           '--epochs':'4','--learning-rate':'1e-5','--iterations':'125',
                           '--checkpoint-interval':'25','--milestone-interval':'125','--seed':'42'}.items():
            assert option(c,flag)==value
        assert '--initialize' in c and '--resume' not in c
        assert p['slot']==i%3
        assert p['environment']['HIP_VISIBLE_DEVICES']==('1' if i%3==0 else '0')
    assert option(plans[1]['command'],'--ppo-update-lock')==option(plans[2]['command'],'--ppo-update-lock')
    subset=m.build_plans(tmp_path,tmp_path/'subset',tmp_path/'cache','server',['world_preview'],125)
    assert subset[0]['slot']==0


def test_desktop_and_strict_resume_are_distinct_from_initializer_transfer(tmp_path):
    m=launcher()
    plan=m.build_plans(tmp_path,tmp_path/'out',tmp_path/'cache','desktop',['world_preview'],1000)[0]
    assert option(plan['command'],'--backend')=='warp'
    assert option(plan['command'],'--epa-horizon')=='96'
    assert '--ppo-update-lock' not in plan['command']
    prior=tmp_path/'prior/world_preview/training/checkpoint.pt'
    prior.parent.mkdir(parents=True)
    prior.write_bytes(b'optimizer')
    resumed=m.build_plans(tmp_path,tmp_path/'resume',tmp_path/'cache','desktop',['world_preview'],25,
                          resume_from=tmp_path/'prior')[0]
    assert option(resumed['command'],'--resume')==str(prior)
    assert '--initialize' not in resumed['command']
    assert resumed['iteration_budget_semantics']=='additional'
    with pytest.raises(ValueError,match='desktop'):
        m.build_plans(tmp_path,tmp_path/'bad',tmp_path/'cache','desktop',['simple_preview'],25)
    with pytest.raises(ValueError,match='positive'):
        m.build_plans(tmp_path,tmp_path/'bad',tmp_path/'cache','server',['world_preview'],0)


def cli(tmp_path,output,extra=(),host='server'):
    bundle=tmp_path/'bundle'
    bundle.mkdir(exist_ok=True)
    (bundle/'bundle.json').write_text(json.dumps({'source_revision':'frozen'}))
    return subprocess.run([sys.executable,str(PATH),'--bundle',str(bundle),'--host',host,
        '--output',str(output),'--reference-cache',str(tmp_path/'cache.pt'),'--dry-run',*extra],
        capture_output=True,text=True)


@pytest.mark.parametrize('host,iterations,names',[
    ('server','125',['simple_masked','simple_preview','world_masked','world_preview']),
    ('desktop','1000',['world_preview'])])
def test_cli_dry_run_defaults_are_reviewable_and_launch_nothing(tmp_path,host,iterations,names):
    output=tmp_path/'planned'
    result=cli(tmp_path,output,host=host)
    assert result.returncode==0,result.stderr
    state=json.loads((output/'status.json').read_text())
    assert state['phase']=='planned' and list(state['runs'])==names
    assert all('pid' not in p and option(p['command'],'--iterations')==iterations for p in state['runs'].values())
    assert not list(output.glob('*/training'))


def test_preflight_and_resume_reject_changed_comparison_contract(tmp_path):
    output=tmp_path/'preflight'
    result=cli(tmp_path,output,['--preflight'])
    assert result.returncode==0,result.stderr
    receipt=json.loads((output/'status.json').read_text())
    assert all(option(p['command'],'--iterations')=='25' for p in receipt['runs'].values())
    receipt['phase']='completed'
    for p in receipt['runs'].values():
        p['phase']='completed'
    good=tmp_path/'good-preflight.json'
    good.write_text(json.dumps(receipt))
    passed=cli(tmp_path,tmp_path/'accepted',['--preflight-report',str(good)])
    assert passed.returncode==0,passed.stderr
    receipt['runs']['world_preview']['comparison_contract']['preview_horizon_s']=0.
    bad=tmp_path/'bad-preflight.json'
    bad.write_text(json.dumps(receipt))
    rejected=cli(tmp_path,tmp_path/'rejected',['--preflight-report',str(bad)])
    assert rejected.returncode!=0 and 'contract differs' in rejected.stderr
    assert not (tmp_path/'rejected').exists()
    for name in receipt['runs']:
        checkpoint=output/name/'training/checkpoint.pt'
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b'saved')
    (output/'status.json').write_text(json.dumps(receipt))
    rejected=cli(tmp_path,tmp_path/'bad-resume',['--resume-from',str(output)])
    assert rejected.returncode!=0 and 'contract differs' in rejected.stderr
    assert not (tmp_path/'bad-resume').exists()
