"""Pin and replay one checkpoint on a chosen reference; publish MuJoCo WASM states."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET
import zipfile


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--request", type=Path, required=True)
    args = p.parse_args()
    spec = json.loads(args.request.read_text())
    root = Path(spec["root"])
    output = Path(spec["output"])
    os.environ["K1_MOTION_ROOT"] = str(root)
    sys.path.insert(0, str(root / "src"))
    import mujoco
    import numpy as np
    import torch
    from k1_motion.contracts import MotionClip
    from k1_motion.robot import K1Model
    from k1_motion.export import export_checkpoint
    from k1_motion.learning import Policy
    from k1_motion.control_validation import replay_clip

    torch.set_num_threads(1)
    checkpoint = Path(spec["checkpoint"])
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    export_checkpoint(checkpoint, output / "actor.pt")
    metadata = json.loads((output / "actor.json").read_text())
    path = Path(spec["reference"]["reference_path"])
    ref_digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if ref_digest != spec["reference"]["reference_sha256"]:
        raise ValueError("Selected reference differs from its catalog receipt")
    original = MotionClip.load(path)
    count = min(len(original.times), round(spec["seconds"] / 0.02) + 1)
    clip = MotionClip(
        original.times[:count],
        {k: v[:count].copy() for k, v in original.values.items()},
        dict(original.metadata),
        original.source_times[:count] if original.source_times is not None else None,
        original.received_times[:count] if original.received_times is not None else None,
    )
    robot = K1Model()
    policy = Policy(output / "actor.pt", robot.signature)
    result = replay_clip(robot, policy, clip, output / "trace.npz")
    trace = np.load(output / "trace.npz")
    actual = np.concatenate(
        [
            np.r_[
                clip.values["root_position"][0],
                clip.values["root_orientation"][0],
                clip.values["joint_position"][0],
            ][None],
            trace["qpos"],
        ]
    )
    desired = np.concatenate(
        [clip.values["root_position"], clip.values["root_orientation"], clip.values["joint_position"]], axis=1
    )
    if not np.isfinite(actual).all():
        raise ValueError("Nonfinite rollout state")
    playback = output / "playback"
    playback.mkdir(exist_ok=True)
    mujoco.mj_saveLastXML(str(playback / "scene.xml"), robot.model)
    tree = ET.parse(playback / "scene.xml")
    xml = tree.getroot()
    compiler = xml.find("compiler")
    meshdir = Path(compiler.get("meshdir", str(robot.model_path.parent / "meshes")))
    compiler.set("meshdir", "")
    compiler.set("texturedir", "")
    assets = []
    for item in xml.findall(".//asset/*"):
        if "file" not in item.attrib:
            continue
        source = Path(item.get("file"))
        if not source.is_absolute():
            source = meshdir / source
        name = hashlib.sha256(source.read_bytes()).hexdigest()[:16] + "-" + source.name
        target = playback / "assets" / name
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(source, target)
        item.set("file", "assets/" + name)
        assets.append("assets/" + name)
    tree.write(playback / "scene.xml", encoding="unicode")
    details = dict(
        run=spec["run"],
        checkpoint_iteration=metadata["checkpoint_iteration"],
        checkpoint_sha256=digest,
        checkpoint_source_revision=metadata.get("source_revision"),
        reference_id=spec["reference"]["id"],
        reference_family=spec["reference"]["family"],
        reference_split=spec["reference"]["split"],
        reference_sha256=ref_digest,
        requested_seconds=spec["seconds"],
        recorded_seconds=(len(actual) - 1) * 0.02,
        original_seconds=float(original.times[-1] - original.times[0]),
        full_original=count == len(original.times),
        execution_errors=0,
        metrics=result,
        scope="CPU closed-loop diagnostic; browser restores recorded states; no promotion",
    )
    for name, states, label in [
        ("states", actual, "Actual controller"),
        ("reference-states", desired, "Requested reference"),
    ]:
        (playback / (name + ".json")).write_text(
            json.dumps(
                dict(
                    fps=50,
                    qpos=states.tolist(),
                    referenceRoot=desired[:, :3].tolist(),
                    actualRoot=actual[:, :3].tolist(),
                    title=f"{spec['run']} · update {metadata['checkpoint_iteration']} · {label}",
                ),
                allow_nan=False,
            )
        )
    for name, episode in [("manifest", "states"), ("reference", "reference-states")]:
        (playback / (name + ".json")).write_text(
            json.dumps(
                dict(model="scene.xml", episode=episode + ".json", assets=assets, metadata=details),
                allow_nan=False,
            )
        )
    with zipfile.ZipFile(output / "playback.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in playback.rglob("*"):
            if path.is_file():
                archive.write(path, str(path.relative_to(playback)))
    (output / "result.json").write_text(json.dumps(details, indent=2, allow_nan=False))
    print(
        json.dumps(
            {k: details[k] for k in ("run", "checkpoint_iteration", "recorded_seconds", "full_original")}
        )
    )


if __name__ == "__main__":
    main()
