"""Additive trajectory diagnostics; proposed screening, not historical acceptance."""
import numpy as np


VERSION = 'k1-trajectory-fidelity-v3-screening'
SCREENING_THRESHOLDS = {'max_root_xy_rmse_m': .25, 'max_root_xy_p95_m': .5}
XY_SCORE_WIDTH_M = .5


def trajectory_fidelity(clip, qpos, *, completed, old_clean_success, initial_root_position=None):
    """Align post-step trace i to target i+1 and charge zero for an unexecuted tail.

    Native replay starts from reference pose/velocity, so its actual initial root
    equals reference[0]. Callers with perturbed initialization must supply the
    measured initial root explicitly for the actual path-length measurement.
    RMS/p95 describe the executed prefix; survival and duration score use the full
    intended recording. Historical clean_success is never changed here.
    """
    target = np.asarray(clip.values['root_position'], dtype=float)
    times = np.asarray(clip.times, dtype=float)
    actual = np.asarray(qpos, dtype=float)
    intervals = np.diff(times)
    count = len(actual)
    intended = len(intervals)
    if (target.ndim != 2 or target.shape != (len(times), 3) or intended < 1
            or not np.isfinite(target).all() or not np.isfinite(times).all()
            or (intervals <= 0).any() or count > intended
            or (count and (actual.ndim != 2 or actual.shape[1] < 3 or not np.isfinite(actual).all()))):
        raise ValueError('Invalid trajectory trace/reference alignment')
    initial = target[0] if initial_root_position is None else np.asarray(initial_root_position, dtype=float)
    if initial.shape != (3,) or not np.isfinite(initial).all():
        raise ValueError('Invalid measured initial root position')
    intended_duration = float(intervals.sum())
    executed_duration = float(intervals[:count].sum())
    error = np.linalg.norm(actual[:, :2]-target[1:count+1, :2], axis=1) if count else np.zeros(0)
    # Longer tails retain ordering for large misses; score is an explicit ranking
    # diagnostic, separate from the proposed stricter root-position gates.
    scores = 1/(1+(error/XY_SCORE_WIDTH_M)**2)
    full_duration_score = float(np.sum(scores*intervals[:count])/intended_duration)
    reference_segments = np.linalg.norm(np.diff(target[:, :2], axis=0), axis=1)
    reference_path = float(reference_segments.sum())
    actual_path = float(np.linalg.norm(np.diff(np.vstack((initial[:2], actual[:, :2])), axis=0), axis=1).sum()) if count else 0.
    rms = float(np.sqrt(np.sum(error**2*intervals[:count])/executed_duration)) if count else None
    p95 = float(np.percentile(error, 95)) if count else None
    clean = bool(old_clean_success and completed and count == intended
                 and rms <= SCREENING_THRESHOLDS['max_root_xy_rmse_m']
                 and p95 <= SCREENING_THRESHOLDS['max_root_xy_p95_m'])
    return {
        'version': VERSION, 'screening_only': True,
        'screening_thresholds': dict(SCREENING_THRESHOLDS),
        'executed_intervals': count, 'intended_intervals': intended,
        'executed_duration_s': executed_duration, 'intended_duration_s': intended_duration,
        'survived_fraction': executed_duration/intended_duration,
        'root_xy_rmse_m': rms, 'root_xy_p95_m': p95,
        'root_xy_max_m': float(error.max()) if count else None,
        'root_xy_final_executed_m': float(error[-1]) if count else None,
        'reference_path_length_m': reference_path,
        'executed_reference_path_length_m': float(reference_segments[:count].sum()),
        'actual_path_length_m': actual_path,
        'actual_path_initialization': 'reference_root_at_t0' if initial_root_position is None else 'supplied_measured_root_at_t0',
        'path_length_ratio': actual_path/reference_path if reference_path > 1e-8 else None,
        'full_reference_duration_xy_score': full_duration_score,
        'xy_score_width_m': XY_SCORE_WIDTH_M,
        'xy_score_formula': 'sum(executed_dt / (1 + (xy_error_m / width_m)^2)) / full_reference_duration; unexecuted tail scores zero',
        'clean_v3': clean,
    }
