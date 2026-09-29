"""Derived stage curves belong to the models, not to the saved configuration.

Two related problems:

* The battery and heating models used to *inject* the zero-power stage the
  optimiser needs into the validated list. That made the model report a
  stage the operator never wrote, so model_dump(exclude_unset=True) -- whose
  whole point is "only what was actually set" -- returned it too, and any
  code writing the model back would have added it to options.json. The
  padded curve is now a plain property, which pydantic keeps out of both
  dumps and the JSON schema.
* The EV model did no ordering validation at all while day_ahead.py took
  charge_stages[-1] as the maximum current, and it prepended the zero stage
  itself, in the optimiser rather than with the model owning the curve.
"""

import pytest
from pydantic import ValidationError

from dao.prog.config.models.devices.battery import BatteryConfig
from dao.prog.config.models.devices.ev import EVConfig
from dao.prog.config.models.devices.heating import HeatingEnabled

BATTERY = {
    "name": "accu",
    "capacity": 10,
    "entity actual level": "sensor.soc",
    "entity set power feedin": "number.feedin",
    "entity set operating mode": "select.mode",
    "dc_to_bat efficiency": 0.95,
    "bat_to_dc efficiency": 0.95,
    "cycle cost": 0.01,
    "charge stages": [{"power": 3000, "efficiency": 0.95}],
    "discharge stages": [{"power": 3000, "efficiency": 0.95}],
}

EV = {
    "name": "auto",
    "capacity": 60,
    "entity position": "device_tracker.auto",
    "entity actual level": "sensor.ev_soc",
    "entity plugged in": "binary_sensor.plugged",
    "charge switch": "switch.charge",
    "entity set charging ampere": "number.amps",
    "charge three phase": True,
    "charge stages": [{"ampere": 6, "efficiency": 0.9}],
    "entity instant start": "switch.instant",
    "entity instant level": "number.instant",
}

HEATING = {
    "heater present": True,
    "adjustment": "power",
    "stages": [{"max_power": 2000, "cop": 4.0}],
}


def battery(**overrides):
    return BatteryConfig(**{**BATTERY, **overrides})


class TestBatteryStagesStayAsWritten:
    def test_the_stored_list_is_exactly_what_was_configured(self):
        config = battery()

        assert [s.power for s in config.charge_stages] == [3000.0]
        assert [s.power for s in config.discharge_stages] == [3000.0]

    def test_exclude_unset_does_not_invent_a_stage(self):
        """The regression that made this worth changing: a dump meant to
        contain only what the operator set came back with an extra stage."""
        config = battery()

        dumped = config.model_dump(by_alias=True, exclude_unset=True)

        assert dumped["charge stages"] == [{"power": 3000.0, "efficiency": 0.95}]

    def test_the_derived_curve_carries_the_zero_stage(self):
        config = battery()

        assert [s.power for s in config.effective_charge_stages] == [0.0, 3000.0]
        assert [s.efficiency for s in config.effective_charge_stages] == [1.0, 0.95]
        assert [s.power for s in config.effective_discharge_stages] == [0.0, 3000.0]

    def test_a_curve_already_starting_at_zero_is_not_padded_twice(self):
        config = battery(
            **{
                "charge stages": [
                    {"power": 0, "efficiency": 1.0},
                    {"power": 3000, "efficiency": 0.95},
                ]
            }
        )

        assert [s.power for s in config.effective_charge_stages] == [0.0, 3000.0]

    def test_the_derived_curve_is_a_copy(self):
        """Callers model_dump it and build MIP variables from it; handing out
        the model's own list would let one run mutate the next."""
        config = battery()

        config.effective_charge_stages.append(None)

        assert len(config.effective_charge_stages) == 2

    def test_the_property_is_not_in_the_json_schema(self):
        """A plain property rather than a computed_field precisely so it
        stays out of the schema the settings UI renders and out of dumps."""
        schema = BatteryConfig.model_json_schema()

        assert "effective_charge_stages" not in schema.get("properties", {})


class TestReducedHours:
    def test_whole_hours_are_accepted(self):
        config = battery(**{"reduced hours": {"22": 1000, "23": 1000}})

        assert config.reduced_hours == {"22": 1000, "23": 1000}

    def test_a_key_that_is_not_a_number_is_refused(self):
        """day_ahead.py does int(key) on these, so this used to raise
        halfway through an optimisation run instead of when saving."""
        with pytest.raises(ValidationError, match="not an hour"):
            battery(**{"reduced hours": {"nacht": 1000}})

    def test_an_hour_outside_the_day_is_refused(self):
        """Worse than a crash: hour 25 matched no interval, so the limit was
        silently never applied and the battery ran at full power all night."""
        with pytest.raises(ValidationError, match="outside 0-23"):
            battery(**{"reduced hours": {"25": 1000}})

    def test_a_negative_power_is_refused(self):
        with pytest.raises(ValidationError, match="cannot be negative"):
            battery(**{"reduced hours": {"22": -1}})

    def test_none_and_empty_stay_valid(self):
        assert battery(**{"reduced hours": None}).reduced_hours is None
        assert battery(**{"reduced hours": {}}).reduced_hours == {}


class TestEvChargeStages:
    def test_an_unsorted_curve_is_refused(self):
        """day_ahead.py takes charge_stages[-1] as the maximum current, so an
        unsorted curve silently capped the car at whatever was listed last."""
        with pytest.raises(ValidationError, match="strictly increasing by ampere"):
            EVConfig(
                **{
                    **EV,
                    "charge stages": [
                        {"ampere": 16, "efficiency": 0.9},
                        {"ampere": 6, "efficiency": 0.9},
                    ],
                }
            )

    def test_duplicate_amperes_are_refused(self):
        with pytest.raises(ValidationError, match="strictly increasing by ampere"):
            EVConfig(
                **{
                    **EV,
                    "charge stages": [
                        {"ampere": 6, "efficiency": 0.9},
                        {"ampere": 6, "efficiency": 0.8},
                    ],
                }
            )

    def test_a_curve_without_any_real_stage_is_refused(self):
        with pytest.raises(ValidationError, match="at least one stage with ampere"):
            EVConfig(**{**EV, "charge stages": [{"ampere": 0, "efficiency": 1.0}]})

    def test_the_derived_curve_carries_the_zero_stage(self):
        config = EVConfig(**EV)

        assert [s.ampere for s in config.effective_charge_stages] == [0.0, 6.0]
        assert config.effective_charge_stages[0].efficiency == 1.0

    def test_the_stored_list_is_untouched(self):
        config = EVConfig(**EV)

        assert [s.ampere for s in config.charge_stages] == [6.0]


class TestHeatingStages:
    def test_the_stored_list_is_untouched(self):
        config = HeatingEnabled(**HEATING)

        assert [s.max_power for s in config.stages] == [2000.0]

    def test_the_derived_curve_carries_the_zero_stage(self):
        config = HeatingEnabled(**HEATING)

        assert [s.max_power for s in config.effective_stages] == [0.0, 2000.0]

    def test_an_empty_list_stays_empty(self):
        """No stages configured means the optimiser builds no stage variables
        at all (S = 0), which is valid for the adjustment modes that do not
        use them. Padding an empty list would turn that into S = 1."""
        config = HeatingEnabled(
            **{**HEATING, "adjustment": "on/off", "stages": []}
        )

        assert config.effective_stages == []

    def test_stages_are_still_required_for_power_adjustment(self):
        with pytest.raises(ValidationError, match="At least one stage is required"):
            HeatingEnabled(**{**HEATING, "adjustment": "power", "stages": []})

    def test_an_unsorted_curve_is_still_refused(self):
        with pytest.raises(ValidationError, match="sorted by max_power"):
            HeatingEnabled(
                **{
                    **HEATING,
                    "stages": [
                        {"max_power": 3000, "cop": 4.0},
                        {"max_power": 1000, "cop": 4.0},
                    ],
                }
            )
