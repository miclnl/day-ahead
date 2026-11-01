"""Zuivere economische componenten, los van solver en afrekenboek."""

from decimal import Decimal, localcontext

from dao.prog.settlement.types import SettlementResult


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


def planning_cost_breakdown(
    *, forecast: SettlementResult, baseline: SettlementResult, battery_wear_eur: Decimal,
    battery_terminal_cost_eur: Decimal, preference_cost_eur: Decimal, other_terminal_cost_eur: Decimal,
) -> dict[str, Decimal | None]:
    """Vergelijk eigen engineboeken; terminalwaarde is modelwaarde, geen kasopbrengst."""
    values = {
        "forecast_invoice_energy_eur": forecast.net_cost,
        "baseline_invoice_energy_eur": baseline.net_cost,
        "battery_wear_eur": battery_wear_eur,
        "battery_terminal_cost_eur": battery_terminal_cost_eur,
        "preference_cost_eur": preference_cost_eur,
        "other_terminal_cost_eur": other_terminal_cost_eur,
    }
    for name, value in values.items():
        if value is None and name in ("forecast_invoice_energy_eur", "baseline_invoice_energy_eur"):
            continue
        if (not isinstance(value, Decimal) or not value.is_finite()
            or (name in ("battery_wear_eur", "preference_cost_eur") and value < 0)):
            raise ValueError(f"{name} must be a finite Decimal with a valid sign")
    with localcontext() as context:
        context.prec = 36
        savings = None if forecast.net_cost is None or baseline.net_cost is None else baseline.net_cost - forecast.net_cost
        values["invoice_energy_savings_eur"] = savings
        values["net_model_advantage_eur"] = (None if savings is None else
            savings - battery_wear_eur - battery_terminal_cost_eur - preference_cost_eur - other_terminal_cost_eur)
    return values
