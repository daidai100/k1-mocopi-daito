import importlib.util
from pathlib import Path
import shutil


def test_concurrent_source_publication_handles_nonempty_destination(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1]/"scripts/freeze_source.py"
    spec = importlib.util.spec_from_file_location("test_source_snapshot", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = tmp_path/"src/k1_motion"
    source.mkdir(parents=True)
    (source/"example.py").write_text("VALUE = 1\n")
    rename = Path.rename

    def publish_other_process(first, second):
        shutil.copytree(first, second)
        return rename(first, second)

    monkeypatch.setattr(Path, "rename", publish_other_process)
    frozen, revision = module.freeze_source(tmp_path)
    assert frozen.name == revision
    assert (frozen/"k1_motion/example.py").read_text() == "VALUE = 1\n"
    assert not list((tmp_path/"artifacts/source-snapshots").glob("k1-source-*"))
