#!/usr/bin/python3
import logging



def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)

    return logger


def validate_sort_by(value_str: str, default: str = 'BIC'):
    valid = ['AICc', 'BIC', 'CV']
    if value_str.strip() == "":
        return default, None
    if value_str in valid:
        return value_str, None
    return None, f"Invalid input, please enter one of {valid}."


def validate_ci(value_str: str, default: float = 0.95):
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
    pass

class ThresholdError(DataError):
    pass

class NoCacheError(DataError):
    pass

