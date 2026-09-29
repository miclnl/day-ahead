"""Validators that keep the battery stage curves usable by day_ahead.py.

day_ahead.py divides by (number of discharge stages - 1) to average the
discharge efficiency, and derives a slope from the SoC gap between
consecutive reduce_power_* entries. Both divide by zero on input the model
used to accept without complaint.
"""

import pytest
from pydantic import ValidationError

from dao.prog.config.models.devices.battery import BatteryConfig

BASE = {
    "name": "accu",
    "capacity": 10,
    "entity actual level": "sensor.soc",
    "entity set power feedin": "number.feedin",
    "entity set operating mode": "select.mode",
    "dc_to_bat efficiency": 0.95,
    "bat_to_dc efficiency": 0.95,
    "cycle cost": 0.01,
    "charge stages": [{"power": 0, "efficiency": 1.0}, {"power": 3000, "efficiency": 0.95}],
    "discharge stages": [{"power": 0, "efficiency": 1.0}, {"power": 3000, "efficiency": 0.95}],
}


def make(**overrides):
    data = {**BASE, **overrides}
    return BatteryConfig(**data)


def test_a_normal_curve_is_accepted():
    battery = make(
        **{
            "charge stages": [{"power": 0, "efficiency": 1.0}, {"power": 3000, "efficiency": 0.95}],
            "discharge stages": [{"power": 0, "efficiency": 1.0}, {"power": 3000, "efficiency": 0.95}],
        }
    )
    assert len(battery.discharge_stages) == 2


def test_only_the_zero_stage_is_rejected():
    """DS would be 1: sum_eff / (DS - 1) divides by zero in day_ahead.py."""
    with pytest.raises(ValidationError, match="power > 0"):
        make(**{"discharge stages": [{"power": 0, "efficiency": 1.0}]})


def test_the_zero_stage_is_added_automatically_when_missing():
    battery = make(**{"discharge stages": [{"power": 3000, "efficiency": 0.95}]})
    assert battery.discharge_stages[0].power == 0.0
    assert len(battery.discharge_stages) == 2


def test_equal_consecutive_powers_are_rejected():
    with pytest.raises(ValidationError, match="strictly increasing"):
        make(
            **{
                "discharge stages": [
                    {"power": 0, "efficiency": 1.0},
                    {"power": 3000, "efficiency": 0.9},
                    {"power": 3000, "efficiency": 0.95},
                ]
            }
        )


def test_unsorted_powers_are_rejected():
    with pytest.raises(ValidationError, match="strictly increasing"):
        make(
            **{
                "discharge stages": [
                    {"power": 3000, "efficiency": 0.9},
                    {"power": 1000, "efficiency": 0.95},
                ]
            }
        )


def test_two_reduce_power_entries_are_accepted():
    battery = make(**{"reduce_power_low_soc": [{"soc": 10, "power": 500}, {"soc": 20, "power": 2000}]})
    assert len(battery.reduce_power_low_soc) == 2


def test_a_single_reduce_power_entry_is_accepted_by_the_model():
    """day_ahead.py itself warns and drops a lone entry; the model must not
    reject it, or that fallback code path can never run."""
    battery = make(**{"reduce_power_low_soc": [{"soc": 10, "power": 500}]})
    assert len(battery.reduce_power_low_soc) == 1


def test_duplicate_soc_in_reduce_power_is_rejected():
    with pytest.raises(ValidationError, match="strictly increasing"):
        make(
            **{
                "reduce_power_high_soc": [
                    {"soc": 90, "power": 2000},
                    {"soc": 90, "power": 500},
                ]
            }
        )


def test_reduce_power_entries_do_not_have_to_be_pre_sorted():
    battery = make(
        **{
            "reduce_power_low_soc": [
                {"soc": 20, "power": 2000},
                {"soc": 10, "power": 500},
            ]
        }
    )
    assert [e.soc for e in battery.reduce_power_low_soc] == [10, 20]
