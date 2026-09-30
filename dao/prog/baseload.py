"""Backward-compatible re-export of the baseload estimator.

The estimator moved to :mod:`dao.forecast.baseload.profile`. This shim keeps
the old import path working for callers that have not moved yet; it is
deleted once nothing references it anymore.
"""

from dao.forecast.baseload.profile import *  # noqa: F401,F403
from dao.forecast.baseload.profile import (  # noqa: F401
    BaseloadOptions,
    BaseloadProfile,
    Sample,
    build_profile,
    is_holiday,
    iter_samples,
    profile_age_days,
    profile_from_file,
    profile_to_dict,
)
