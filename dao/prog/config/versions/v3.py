"""
Configuration schema version 3.

Changes from v2:
- Every solar installation names its forecasting model explicitly
  (``model``: physical | ml | auto) instead of the boolean
  ``ml_prediction``, which could only say "the old XGBoost model" or "the
  old physics approximation" and had no way to express "let the backtest
  decide".

All other fields are inherited from ConfigurationV2.
"""

from typing import Literal

from .v2 import ConfigurationV2


class ConfigurationV3(ConfigurationV2):
    """Configuration schema version 3."""

    config_version: Literal[3] = 3
