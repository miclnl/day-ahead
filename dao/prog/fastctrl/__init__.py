"""Fast control: the realtime feedback layer on top of the day-ahead plan.

Module layout, in dependency order:

``plan``
    The hand-off artefact between the two layers. The optimizer writes it after
    every successful run; the fast layer reads it. Pure data, no I/O besides
    reading and writing a single JSON file.

``policy``
    The decision logic. A pure function of (plan, measurements, controller
    state) with no Home Assistant, database or configuration dependency, so it
    can be unit tested and replayed in the backtest exactly as it runs in
    production.

``runner``
    The process side: reads Home Assistant, calls the policy, writes the
    setpoint, persists controller state across restarts.

``simulate``
    Replays historical measurements through the very same policy to quantify
    the saving before you switch the layer on.
"""

from .plan import (
    FAST_PLAN_FILE,
    BatteryPlanStep,
    BatterySpec,
    FastPlan,
    PlanInterval,
    load_plan,
    write_plan,
)
from .policy import (
    ControllerState,
    Decision,
    FastControlPolicy,
    Measurement,
    PolicyLimits,
)

__all__ = [
    "FAST_PLAN_FILE",
    "BatteryPlanStep",
    "BatterySpec",
    "ControllerState",
    "Decision",
    "FastControlPolicy",
    "FastPlan",
    "Measurement",
    "PlanInterval",
    "PolicyLimits",
    "load_plan",
    "write_plan",
]
