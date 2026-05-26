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


class DataError(RuntimeError):
    pass

class ThresholdError(DataError):
    pass

class NoCacheError(DataError):
    pass

