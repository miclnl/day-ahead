"""Zuivere economische componenten, los van solver en afrekenboek."""

from decimal import Decimal, localcontext


def battery_wear_cost(
    *, dc_charge_kwh: Decimal, dc_discharge_kwh: Decimal,
    eur_per_dc_bus_throughput_kwh: Decimal,
) -> Decimal:
    """Tel beide DC-buszijden; geen capaciteits- of rendementsomrekening."""
    for name, value in (
        ("dc_charge_kwh", dc_charge_kwh),
        ("dc_discharge_kwh", dc_discharge_kwh),
        ("eur_per_dc_bus_throughput_kwh", eur_per_dc_bus_throughput_kwh),
    ):
        if not isinstance(value, Decimal) or not value.is_finite() or value < 0:
            raise ValueError(f"{name} must be a finite non-negative Decimal")
    with localcontext() as context:
        context.prec = 36
        return (dc_charge_kwh + dc_discharge_kwh) * eur_per_dc_bus_throughput_kwh
