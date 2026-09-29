"""Forecasting of household baseload and PV production.

Five units with one job each: ``history`` reads Home Assistant statistics,
``baseload`` and ``pv`` forecast the two quantities the optimizer needs,
``weather`` fetches and stores the weather inputs, ``evaluate`` measures how
good the forecasts were. Nothing in here imports the optimizer or the
reporting module; the adapters that connect them live in ``dao/prog``.
"""
