import importlib.util
from pathlib import Path


def test_kneeling_diagnostic_does_not_count_sitting_with_hands_on_knees():
    path = Path(__file__).parents[1] / "scripts/analyze_family_rejections.py"
    spec = importlib.util.spec_from_file_location("family_rejection_diagnostic", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    def category(text):
        return module.support_category({"family": "sit_or_kneel", "annotations": [text]})
    assert category("sitting with hands on knees") == "floor_sitting_or_unspecified"
    assert category("sitting on a chair with hands on knees") == "seated_furniture"
    assert category("transition from standing to kneeling") == "kneeling_no_furniture_mentioned"
    assert category("kneeling supported by a chair") == "kneeling_with_furniture"
