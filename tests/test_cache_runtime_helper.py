"""Cache rebinding must ignore only an unused runtime helper, never preprocessing.

Test-first: a helper-only source difference is reusable without FK; any retained
call, import change, definition-time side effect, or MotionLibrary edit rejects.
"""
import ast
from pathlib import Path
import shutil

import pytest
import torch

import k1_motion.reference_cache as cache
from test_training import make_library


def without_helper(source):
    tree = ast.parse(source)
    tree.body = [node for node in tree.body if not (
        isinstance(node, ast.FunctionDef) and node.name == 'declared_actuator_contract')]
    return ast.unparse(tree)


def test_only_unreferenced_runtime_actuator_helper_is_excluded():
    source = (Path(cache.__file__).parent/'learning.py').read_text()
    before = without_helper(source)
    assert cache._preprocessing_ast(source) == cache._preprocessing_ast(before)
    # Imports remain part of the proof, including imports useful only at runtime.
    assert cache._preprocessing_ast(before+'\nimport fractions\n') != cache._preprocessing_ast(before)
    changed = source.replace('class MotionLibrary:', 'class MotionLibrary:\n    preprocessing_change = True')
    assert cache._preprocessing_ast(changed) != cache._preprocessing_ast(source)
    changed = source.replace('class MotionLibrary:',
        'class MotionLibrary:\n    construction_dependency = declared_actuator_contract({}, {})')
    with pytest.raises(ValueError, match='runtime helper'):
        cache._preprocessing_ast(changed)
    changed = source.replace('def declared_actuator_contract(settings, physics_contract):',
        '@side_effect\ndef declared_actuator_contract(settings, physics_contract):')
    with pytest.raises(ValueError, match='runtime helper'):
        cache._preprocessing_ast(changed)


def test_cache_rebind_runtime_helper_does_not_reconstruct_references(tmp_path, monkeypatch):
    from k1_motion.learning import MotionLibrary
    robot, directory = make_library(tmp_path)
    package = Path(cache.__file__).parent
    old = tmp_path/'old-package'
    old.mkdir()
    for name in cache.PREPROCESSING_FILES:
        shutil.copy2(package/name, old/name)
    (old/'learning.py').write_text(without_helper((old/'learning.py').read_text()))
    original = tmp_path/'original.pt'
    current_file = cache.__file__
    # The constructor is identical; create the small synthetic fixture bound to
    # its helper-free source package, then prove the rebind performs no rebuild.
    monkeypatch.setattr(cache, '__file__', str(old/'reference_cache.py'))
    cache.build_reference_cache(directory, original, robot)
    before = cache.load_reference_cache(original, directory, robot, 'cpu')
    monkeypatch.setattr(cache, '__file__', current_file)

    def must_not_construct(*args, **kwargs):
        raise AssertionError('Rebinding must not rebuild reference/FK data')

    monkeypatch.setattr(MotionLibrary, '__init__', must_not_construct)
    result = cache.rebind_reference_cache(original, tmp_path/'rebound.pt', old, directory, robot)
    assert result['tensor_transformations'] == 0
    assert 'declared_actuator_contract' in result['source_proof']
    after = cache.load_reference_cache(tmp_path/'rebound.pt', directory, robot, 'cpu')
    for name in before.values:
        torch.testing.assert_close(before.values[name], after.values[name], atol=0, rtol=0)
