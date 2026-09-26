"""Derive the Isaac collision shapes from the authoritative K1 MJCF contract."""

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from .math3d import rotation


def shared_collision_urdf(root, model_path):
    root, model_path = Path(root), Path(model_path)
    original = model_path.with_suffix(".urdf")
    xml, urdf = ET.parse(model_path), ET.parse(original)
    mesh_dir = model_path.parent / xml.getroot().find("compiler").get("meshdir", "")
    meshes = {m.get("name"): m for m in xml.findall(".//asset/mesh")}
    counts = {}
    for body in xml.findall(".//body"):
        link = urdf.find(f".//link[@name='{body.get('name')}']")
        if link is None:
            continue
        for node in list(link.findall("collision")):
            link.remove(node)
        count = 0
        for geom in body.findall("geom"):
            if geom.get("contype", "1") == "0" and geom.get("conaffinity", "1") == "0":
                continue
            node = ET.SubElement(link, "collision", name=f"contract_collision_{count}")
            rpy = rotation(np.fromstring(geom.get("quat", "1 0 0 0"), sep=" ")).as_euler("xyz")
            ET.SubElement(node, "origin", xyz=geom.get("pos", "0 0 0"), rpy=" ".join(map(str, rpy)))
            shape = ET.SubElement(node, "geometry")
            size = np.fromstring(geom.get("size", ""), sep=" ")
            kind = geom.get("type", "sphere")
            if kind == "box":
                ET.SubElement(shape, "box", size=" ".join(map(str, 2 * size)))
            elif kind == "cylinder":
                ET.SubElement(shape, "cylinder", radius=str(size[0]), length=str(2 * size[1]))
            elif kind == "sphere":
                ET.SubElement(shape, "sphere", radius=str(size[0]))
            elif kind == "mesh":
                mesh = meshes[geom.get("mesh")]
                ET.SubElement(
                    shape,
                    "mesh",
                    filename=str((mesh_dir / mesh.get("file")).resolve()),
                    scale=mesh.get("scale", "1 1 1"),
                )
            else:
                raise ValueError(f"Unimplemented collision conversion: {kind}")
            count += 1
        counts[body.get("name")] = count
    # Preserve upstream visuals but make paths independent of the cache directory.
    for mesh in urdf.findall(".//mesh"):
        path = Path(mesh.get("filename"))
        if not path.is_absolute():
            mesh.set("filename", str((original.parent / path).resolve()))
    content = ET.tostring(urdf.getroot(), encoding="utf-8", xml_declaration=True)
    digest = hashlib.sha256(content).hexdigest()
    directory = root / "artifacts/simulator-assets" / digest
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "K1_shared_collision.urdf"
    if not path.exists():
        path.write_bytes(content)
        (directory / "receipt.json").write_text(
            json.dumps(
                {
                    "source_urdf": str(original),
                    "collision_source": str(model_path),
                    "sha256": digest,
                    "collision_counts": counts,
                    "inertials_and_joint_frames": "unchanged from pinned official URDF",
                },
                indent=2,
            )
            + "\n"
        )
    return path
