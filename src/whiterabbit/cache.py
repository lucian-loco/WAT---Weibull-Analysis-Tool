#!/usr/bin/env python3
import time
from functools import wraps

class ExpiringCacheMixin:
    """ Mixin for caching the results of a method for a certain period of time. """
    def __init__(self, expiration_period):
        # Time of the last call to the cache
        self._last_call_time = 0
        # Default expiration period for the cache (in seconds)
        self._default_expiration_period = 5


    def decorate(expiration_period=None):
        """ Decorator for caching the results of a method for a certain period of time. """
        def decorator(func):
            @wraps(func)
            def wrapper(self, *args, **kwargs):
                nonlocal expiration_period

                if expiration_period is None:
                    expiration_period = self._default_expiration_period

                now = time.time()
                
                if (now - self._last_call_time) > expiration_period:
                    self._update_cache()
                    self._last_call_time = now

                return func(self, *args, **kwargs)

            return wrapper
        return decorator
    

    def set_expiration_period(self, expiration_period):
        """ Set the default expiration period for the cache (in seconds). """
        self._default_expiration_period = expiration_period


    def _update_cache(self):
        """ You should implement this method in the child class. """
        raise NotImplemented