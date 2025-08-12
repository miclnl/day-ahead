import json
from typing import List, Tuple, Dict


def parse_curve(curve_str: str) -> List[Tuple[float, float]]:
    """Parse a curve string like '0:85,10:90,50:95,100:80' into sorted (x,y) pairs.
    x typically is SoC (%), y is value (efficiency % or power %).
    """
    if not curve_str:
        return []
    points: List[Tuple[float, float]] = []
    for part in curve_str.split(','):
        if not part.strip():
            continue
        try:
            x_str, y_str = part.split(':')
            x = float(x_str.strip())
            y = float(y_str.strip())
            points.append((x, y))
        except Exception:
            continue
    points.sort(key=lambda p: p[0])
    return points


def interp(points: List[Tuple[float, float]], x: float) -> float:
    """Linear interpolation of piecewise defined curve points.
    If outside range, clamp to nearest endpoint.
    """
    if not points:
        return 0.0
    if x <= points[0][0]:
        return points[0][1]
    if x >= points[-1][0]:
        return points[-1][1]
    for i in range(1, len(points)):
        x0, y0 = points[i - 1]
        x1, y1 = points[i]
        if x0 <= x <= x1:
            if x1 == x0:
                return y0
            t = (x - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)
    return points[-1][1]


def average_value_over_range(points: List[Tuple[float, float]], start_x: float, end_x: float, step: float = 5.0) -> float:
    """Average interpolated y over [start_x, end_x] with a given step.
    Returns 0.0 if points is empty.
    """
    if not points:
        return 0.0
    if end_x < start_x:
        start_x, end_x = end_x, start_x
    samples: List[float] = []
    x = start_x
    while x <= end_x + 1e-9:
        samples.append(interp(points, x))
        x += step
    return sum(samples) / len(samples) if samples else 0.0


def load_calibration(path: str) -> Dict[str, Dict[str, float]]:
    """Load battery calibration dict from JSON file.
    Structure: { battery_name: { 'charge_factor': float, 'discharge_factor': float } }
    """
    try:
        with open(path, 'r') as f:
            data = json.load(f)
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return {}


