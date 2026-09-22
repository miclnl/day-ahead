"""
History/data retention configuration models.
"""

from pydantic import BaseModel, Field, ConfigDict


class HistoryConfig(BaseModel):
    """History and data retention settings."""

    save_days: int = Field(
        alias="save days",
        default=7,
        ge=1,
        description="Number of days to retain historical data",
        json_schema_extra={
            "x-help": "Number of days to retain optimization history in database. Older data is automatically cleaned up. Longer retention enables better trend analysis but increases database size. Minimum 1 day.",
            "x-unit": "days",
            "x-ui-section": "History",
            "x-validation-hint": "Must be >= 1, typical 7-30 days",
        },
    )

    forecast_days: int = Field(
        alias="forecast days",
        default=60,
        ge=7,
        description="Number of days of forecast history kept for accuracy reporting",
        json_schema_extra={
            "x-help": "The forecast archive records what was predicted and how far ahead, "
            "so the forecast error can be measured afterwards. It holds one row "
            "per variable, moment and lead time bucket, which keeps it bounded "
            "no matter how often the optimizer runs: roughly 10 MB at the "
            "default of 60 days. Reduce it on a machine with limited storage, "
            "such as a Home Assistant Yellow on eMMC.",
            "x-unit": "days",
            "x-ui-section": "History",
            "x-validation-hint": "Must be >= 7, typical 30-90 days",
        },
    )

    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
        json_schema_extra={
            "x-ui-group": "Reporting",
            "x-icon": "database-clock",
            "x-order": 15,
            "x-help": """# History & Data Retention

Control how long optimization history is retained in the database.

## What Gets Stored

- Optimization results (costs, schedules)
- Price data (day-ahead, tariffs)
- Device schedules (battery, EV, heating)
- Solar production forecasts
- Baseload consumption data

## Retention Guidelines

- **7 days**: Minimal retention, recent data only
- **14 days**: Two weeks for comparison
- **30 days**: Monthly trends and analysis
- **90+ days**: Long-term analysis (larger database)

## Database Growth

- More retention = larger database
- Typical: ~1-5 MB per day (depends on devices)
- Monitor database size if using SQLite
- Consider periodic backups

## Tips

- Start with 7-14 days, increase if needed
- Check database size regularly
- Old data cleaned up automatically
- Increase for detailed cost analysis
""",
            "x-docs-url": "https://github.com/miclnl/day-ahead/wiki/History-Configuration",
        },
    )
