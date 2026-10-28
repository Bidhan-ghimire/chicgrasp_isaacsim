"""Edit this demonstration; recorder.py consumes this exact builder.

Commands are 10 Hz, [base x, y, z, left-open, right-open]. No Kit imports.
The default matches attachment_capture.frozen_candidate_actions from the
preserved source. Its world-to-base rotation is identity (DESIGN_BASIS).
"""
WAYPOINTS = {
    'A': (0.65, -0.16, 0.2155),
    'B': (0.65, -0.16, 0.11130000000000001),
    'C': (0.650011448690015, -0.11144560745023943, 0.11130000000000001),
    'D': (0.650011448690015, -0.11144560745023943, 0.10041283641442127),
    'L': (0.650011448690015, -0.11144560745023943, 0.20041283641442126),
}

def build_demo(position, options=None):
    import numpy as np
    options = options or {}
    unknown = set(options) - {'lift_extra_m'}
    if unknown:
        raise ValueError('Unknown script options: ' + repr(sorted(unknown)))
    extra = float(options.get('lift_extra_m', 0.0))
    if not np.isfinite(extra) or not 0 <= extra <= 0.02:
        raise ValueError('Example lift_extra_m is bounded to 0..0.02 metres.')
    target = {k: np.asarray(v, dtype=np.float64) for k, v in WAYPOINTS.items()}
    target['L'] = target['L'].copy()
    target['L'][2] += extra
    rows = []
    for i in range(200):
        if i < 50: xyz = target['A']
        elif i < 90: xyz = target['A'] + (target['B'] - target['A']) * ((i - 49) / 40)
        elif i < 110: xyz = target['B'] + (target['C'] - target['B']) * ((i - 89) / 20)
        elif i < 120: xyz = target['C'] + (target['D'] - target['C']) * ((i - 109) / 10)
        elif i < 140: xyz = target['D']
        elif i < 180: xyz = target['D'] + (target['L'] - target['D']) * ((i - 139) / 40)
        else: xyz = target['L']
        rows.append(np.r_[xyz, (1, 1) if i < 120 else (0, 0)].astype(np.float32))
    # Preserve the original float32 -> float64 translation -> float32 arithmetic.
    result = np.asarray(rows, dtype=np.float32).astype(np.float64)
    delta = np.asarray(position['authored_xyz_m'], dtype=np.float64) - [0.65, 0, 0.8155]
    # XYZ translation supports raised custom scenes. Edit the trajectory itself
    # for a different task or orientation; this does not rotate the fixed wrist.
    result[:, :3] += delta
    return result.astype(np.float32)

if __name__ == '__main__':
    from pipeline.runtime import main
    main('script')
