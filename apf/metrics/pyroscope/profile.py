import functools
import os
import pyroscope

def profile(func):
    """ Creates a Pyroscope context for the function to execute """
    @functools.wraps(func)
    def pyroscope_context(func):
        if os.getenv("USE_PROFILING"):
            with pyroscope.tag_wrapper({ "function": func.__name__ }):
                func()

        else:
            func()

    return pyroscope_context