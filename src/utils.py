"""
General purpose utility
"""

import time

# Module Vars
_debug = True
_start_times = {}
_times = {}


# Timer methods
# TODO: these would be better suited to be wrapped in a class
# TODO: Make a "history" class similar to TF's
def start_timer(name):
    """
    records the start time for 'name' for retrieval
    """
    _start_times[name] = time.time()

def add_timer(name):
    delta_t = time.time() - _start_times.get(name)
    if name not in _times:
        _times[name]=0
    _times[name] += delta_t

def end_timer(name):
    """
    calculates the deltatime between now and the start_time call for 'name'
    will throw a dict exception if start_time hasn't been called first
    """
    t_now = time.time()
    _times[name] = t_now - _start_times.get(name)


def get_time(name, format="seconds"):
    """
    returns the deltatime for 'name'
    Will throw an exception if start and end timer haven't been called
    """
    t = _times.get(name)
    if format == "miliseconds" or format =="ms":
        t *= 1000
    return t

def get_times():
    return _times.items()


# Debugging Methods

def debug_print(*args):
    """
    As a replacement for the print command, this allows printing to be toggled on and off using the global debug flag
    """
    if _debug: print(*args)


def set_debug(bool):
    global _debug
    _debug = bool
