import functools


def compose(*functions):
    "Returns a composite function based on the provided functions"
    return functools.reduce(lambda f, g: lambda x: f(g(x)), functions, lambda x: x)
