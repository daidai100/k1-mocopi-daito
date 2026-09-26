"""Deterministic behavioral checkpoint beam, with independent per-run champions."""


def candidate_beam(comparisons, width=4):
    def rank(key):
        result = comparisons[key]
        stats = result["all"]
        return (-stats["clean"], -sum(f["clean"] > 0 for f in result["by_family"].values()),
                stats.get("collision_trials", 54), -stats["completed"],
                result["training_exposure"].get("checkpoint_transitions") or 0, key)
    ordered = sorted((k for k, v in comparisons.items() if not v.get("execution_errors")), key=rank)
    best = {}
    for key in ordered:
        best.setdefault(key.split("/")[0], key)
    return {"width": width, "selected": ordered[:width], "best_per_run": best,
            "ranking": "clean trials, clean family breadth, fewer collision trials, completions, lower exposure",
            "selection_panel_not_unseen_test": True, "automatically_promoted": False,
            "deletion_policy": "No checkpoints deleted; numbered milestones and terminal checkpoints remain immutable"}
