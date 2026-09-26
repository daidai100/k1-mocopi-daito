"""Freeze small controller sources so TorchScript never inspects a changing file."""

import hashlib
import errno
from pathlib import Path
import shutil
import tempfile


def freeze_source(project_root, source_directory=None):
    root = Path(project_root)
    source = Path(source_directory) if source_directory else root / "src/k1_motion"
    contents = {str(p.relative_to(source)): p.read_bytes() for p in sorted(source.rglob("*"))
                if p.is_file() and p.suffix in (".py", ".cpp")}
    digest = hashlib.sha256()
    for name, content in contents.items():
        digest.update(name.encode())
        digest.update(content)
    revision = digest.hexdigest()
    cache = root / "artifacts/source-snapshots"
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / revision
    if not target.is_dir():
        staging = Path(tempfile.mkdtemp(prefix="k1-source-", dir=cache))
        package = staging / "k1_motion"
        package.mkdir()
        for name, content in contents.items():
            (package / name).parent.mkdir(parents=True, exist_ok=True)
            (package / name).write_bytes(content)
        try:
            staging.rename(target)
        except OSError as error:
            # Concurrent launchers can publish the same immutable snapshot.
            # Linux rename(nonempty-directory) reports ENOTEMPTY, not EEXIST.
            if error.errno not in (errno.EEXIST, errno.ENOTEMPTY) or not target.is_dir():
                raise
            shutil.rmtree(staging)
    return target, revision
