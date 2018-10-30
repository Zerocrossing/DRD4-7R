"""
General purpose utility
"""

#Module Vars
debug = True

#Debugging Methods

def debug_print(*args):
    """
    As a replacement for the print command, this allows printing to be toggled on and off using the global debug flag
    """
    if debug: print(*args)

def set_debug(bool):
    global debug
    debug = bool
