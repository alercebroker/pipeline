import functools


def compose(*functions):
    """Returns a composite function based on the provided functions"""

    def pack(x):
        return x if isinstance(x, tuple) else (x,)

    return functools.reduce(
        lambda f, g: lambda *x: f(*pack(g(*pack(x)))), functions, lambda *x: x
    )
