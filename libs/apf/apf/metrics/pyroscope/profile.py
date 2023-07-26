import functools
import os
import pyroscope


def profile(func):
    """Creates a Pyroscope context for the function to execute"""

    @functools.wraps(func)
    def pyroscope_context(*args, **kwargs):
        if bool(os.getenv("USE_PROFILING")):
            with pyroscope.tag_wrapper({"function": func.__name__}):
                func(*args, **kwargs)

        else:
            func(*args, **kwargs)

    return pyroscope_context
