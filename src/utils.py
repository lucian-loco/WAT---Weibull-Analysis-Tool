#!/usr/bin/python3
"""
utils.py
=========

Shared utility module for the Weibull reliability analysis toolset,
providing centralized logging configuration, user-input validation
helpers, and custom exception types.

Contents
--------
- `LocalTimeFormatter` / `get_logger`: A consistent, timezone-aware
  (Europe/Zurich) logging setup used across all modules in this project,
  so log timestamps reflect local time regardless of server timezone.
- `validate_*` functions: Input validation/parsing helpers for the
  interactive CLI prompts (used by `weibull_user_input.py`'s
  `ask_sort_by`, `ask_ci`, and related functions), each following a
  uniform (value, error_message) return contract so the caller can loop
  until valid input is provided.
- `DataError`, `ThresholdError`, `NoCacheError`: Custom exception
  hierarchy used throughout the data-access and analysis modules
  (`data_weibull.py`, `weibull_analysis.py`, `weibull_evaluation.py`,
  `weibull_forecast.py`) to distinguish data-availability problems from
  generic runtime errors.

Author: Lucian Groha
"""
import logging
import datetime
import zoneinfo



class LocalTimeFormatter(logging.Formatter):
    """
    Logging formatter that renders the log record's timestamp in the Europe/Zurich timezone,
    regardless of the host machine's local timezone or UTC settings.

    Overrides `format` to compute `record.asctime` from the record's creation time (`record.created`, a Unix timestamp)
    converted to Europe/Zurich, formatted as 'YYYY-MM-DD HH:MM:SS', before delegating to
    the standard `logging.Formatter.format` for the rest of the message construction.
    """
    def format(self, record):
        tz = zoneinfo.ZoneInfo('Europe/Zurich')
        dt = datetime.datetime.fromtimestamp(record.created, tz=tz)
        record.asctime = dt.strftime('%Y-%m-%d %H:%M:%S')
        return super().format(record)


def get_logger(name):
    """
    Create (or retrieve) a module-level logger configured with a Europe/Zurich-timestamped console handler.

    Ensures each named logger has exactly one `StreamHandler` attached (avoiding duplicate handlers/log lines
    on repeated calls, e.g. when a module is reloaded or imported multiple times), sets the level to INFO,
    and disables propagation to the root logger.

    Parameters
    ----------
    name : str
        Logger name, typically `__name__` of the calling module, so log output can be traced back to its source module.

    Returns
    -------
    logging.Logger
        A configured logger instance ready for use, with format '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        and Europe/Zurich local timestamps.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = LocalTimeFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def validate_sort_by(value_str: str, default: str = 'BIC'):
    """
    Validate and normalize user input for the model-selection sort criterion prompt.

    Parameters
    ----------
    value_str : str
        Raw user input string.
    default : str, optional
        Value to return if `value_str` is empty/whitespace-only (default: 'BIC').

    Returns
    -------
    tuple
        (value, error_message): `value` is `default` (if input was blank), the validated input
        (if one of 'AICc', 'BIC', 'CV'), or None (if invalid). `error_message` is None on success, or a user-facing
        string describing the valid options on failure.
    """
    valid = ['AICc', 'BIC', 'CV']
    if value_str.strip() == "":
        return default, None
    if value_str in valid:
        return value_str, None
    return None, f"Invalid input, please enter one of {valid}."


def validate_ci(value_str: str, default: float = 0.95):
    """
    Validate and normalize user input for a confidence-level prompt.

    Parameters
    ----------
    value_str : str
        Raw user input string.
    default : float, optional
        Value to return if `value_str` is empty/whitespace-only (default: 0.95).

    Returns
    -------
    tuple
        (value, error_message): `value` is `default` (if input was blank), the parsed float (if in the range [0, 1)),
        or None (if invalid or out of range). `error_message` is None on success, or a user-facing string on failure.
        Note: 0 is treated as a valid input, meaning "no confidence interval";
        1 is excluded as an invalid confidence level.
    """
    if value_str.strip() == "":
        return default, None
    try:
        v = float(value_str)
        if 0 <= v < 1:
            return v, None
        return None, "Please enter a value strictly between 0 and 1 or 0 for no confidence interval."
    except ValueError:
        return None, "Invalid input, please enter a number (e.g. 0.95)."


def validate_type(value_str: str, default: str = 'Failure Probability'):
    """
    Validate and normalize user input for the plot-type prompt (failure probability / CDF vs. survival probability / SF),
    mapping several accepted synonyms to a single canonical internal value.

    Parameters
    ----------
    value_str : str
        Raw user input string.
    default : str, optional
        Value to return if `value_str` is empty/whitespace-only (default: 'Failure Probability').

    Returns
    -------
    tuple
        (value, error_message): `value` is 'CDF' (if input, or the default, is one of 'Failure Probability', 'Failure',
        'CDF'), 'SF' (if input, or the default, is one of 'Survival Probability', 'Survival', 'SF'), or None
        (if invalid). `error_message` is None on success, or a user-facing string listing valid options on failure.

    Notes
    -----
    If `value_str` is blank, `default` is returned as-is without being mapped to 'CDF'/'SF' — only non-blank recognized
    inputs get normalized to the canonical form.
    """
    valid = ['Failure Probability', 'Failure', 'CDF', 'Survival Probability', 'Survival', 'SF']
    if value_str.strip() == "":
        return default, None
    if value_str in valid:
        if value_str.strip() in ['Failure Probability', 'Failure', 'CDF']:
            value_str = 'CDF'
            return value_str, None
        else:
            value_str = 'SF'
            return value_str, None
    return None, f"Invalid input, please enter one of {valid}."


def validate_fc(value_str: str, default: list[float] = None):
    """
    Validate and parse user input for a comma-separated list of forecast time horizons.

    Parameters
    ----------
    value_str : str
        Raw user input string, expected as comma-separated numbers (e.g. "365, 730").
    default : list[float], optional
        Values to return if `value_str` is empty/whitespace-only. If not provided,
        defaults to [180.0, 365.0, 730.0, 1095.0].

    Returns
    -------
    tuple
        (values, error_message): `values` is `default` (if input was blank), a list of parsed floats
        (if all values are numeric and >= 1), or None (if parsing failed, the list is empty, or any value is < 1).
        `error_message` is None on success, or a user-facing string describing the problem on failure.
    """
    if default is None:
        default = [180.0, 365.0, 730.0, 1095.0]
    if value_str.strip() == "":
        return default, None
    try:
        values = [float(v.strip()) for v in value_str.split(',') if v.strip()]
        if not values:
            return None, "Please enter at least one forecast time."
        if any(v < 1 for v in values):
            return None, "All forecast times must be ≥ 1 day."
        return values, None
    except ValueError:
        return None, "Invalid input — use numbers separated by commas (e.g. 365, 730)."


class DataError(RuntimeError):
    """
    Base exception for data-availability and data-quality problems encountered while fetching or validating Weibull
    reliability data (e.g. no matching parts found, missing required columns, insufficient failure records).
    """
    pass

class ThresholdError(DataError):
    """
    Raised when failure-count or distinct-failure-time thresholds used to select Weibull analysis candidates are invalid
    (e.g. non-integer, below the required minimum) or otherwise cannot be satisfied.
    """
    pass

class NoCacheError(DataError):
    """
    Raised when the in-memory Weibull data cache is required but empty and an on-demand refresh attempt
    also fails to populate it.
    """
    pass

