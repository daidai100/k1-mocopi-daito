"""An unreviewed checkpoint must never authorize another training block."""
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))


def test_receipt_binds_checkpoint_and_rejects_stale_approval(tmp_path):
    from k1_motion.milestone_review import review_checkpoint
    checkpoint=tmp_path/'checkpoint-000125.pt'
    checkpoint.write_bytes(b'checkpoint identity')
    reviews=tmp_path/'reviews'
    reviews.mkdir()
    receipt=reviews/(checkpoint.stem+'.json')
    receipt.write_text(json.dumps(dict(checkpoint_sha256='stale',continue_training=True)))
    with pytest.raises(ValueError,match='checkpoint'):
        review_checkpoint(checkpoint,reviews,timeout_s=1)
    receipt.write_text(json.dumps(dict(checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        continue_training=False,reason='lost clean motion')))
    assert review_checkpoint(checkpoint,reviews,timeout_s=1)['continue_training'] is False
    receipt.unlink()
    result=review_checkpoint(checkpoint,reviews,timeout_s=.02)
    assert result['continue_training'] is False and result['reason']=='review_timeout'


def test_real_trainer_stops_after_rejected_milestone_preserves_checkpoint(tmp_path):
    import threading
    import time
    import torch
    from test_training import make_library
    from k1_motion.tracking_env import TrackerEnv
    from k1_motion.learning import TrainConfig,train
    robot,library=make_library(tmp_path)
    output=tmp_path/'training'
    reviews=tmp_path/'reviews'
    reviews.mkdir()
    def reviewer():
        path=output/'checkpoint-000001.pt'
        deadline=time.monotonic()+15
        while not path.exists() and time.monotonic()<deadline:
            time.sleep(.01)
        if path.exists():
            body=dict(checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                continue_training=False,reason='regression')
            partial=reviews/'receipt.partial'
            partial.write_text(json.dumps(body))
            partial.rename(reviews/'checkpoint-000001.json')
    worker=threading.Thread(target=reviewer,daemon=True)
    worker.start()
    torch.set_num_threads(1)
    env=TrackerEnv(library,2,'cpu','mujoco_cpp',history=3,physics_options={'workers':2})
    cfg=TrainConfig(stage='student',iterations=3,horizon=4,epochs=1,minibatch=8,
        hidden_sizes=(16,8),bc_weight=0,evaluation_interval=0,checkpoint_interval=1,
        milestone_interval=1,milestone_review_directory=str(reviews),milestone_review_timeout_s=20)
    try:
        result=train(env,output,cfg)
        assert result['iterations']==1
        assert result['stop_reason']=='milestone_review:regression'
        assert result['finite_updates'] and result['checkpoint_reload_max_error']==0
        assert (output/'checkpoint-000001.pt').is_file()
        assert not (output/'checkpoint-000002.pt').exists()
    finally:
        env.close()
        worker.join(timeout=1)
