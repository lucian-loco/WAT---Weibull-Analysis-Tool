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


class DataError(RuntimeError):
    pass

class ThresholdError(DataError):
    pass

class NoCacheError(DataError):
    pass

