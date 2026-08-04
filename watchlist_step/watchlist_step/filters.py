def satisfies_filter(values: dict, type: str, params: dict) -> bool:
    match type:
        case "constant":
            return constant(values, **params)
        case "and":
            return _all(values, **params)
        case "or":
            return _any(values, **params)
        case "no filter":
            return True
        case _:
            raise Exception(f"Invalid filter type: {type}")


def constant(values: dict, field: str, constant, op: str) -> bool:
    value = values[field]
    # User-supplied thresholds can arrive as JSON strings (e.g. "0.5"); the
    # ordering ops need numeric operands, so coerce both sides.
    if op != "eq":
        value, constant = float(value), float(constant)
    match op:
        case "less":
            return value < constant
        case "less eq":
            return value <= constant
        case "greater":
            return value > constant
        case "greater eq":
            return value >= constant
        case "eq":
            return value == constant
        case _:
            raise Exception(f"Invalid value for filter.params.op: {op}")


def _all(values: dict, filters: list[dict]) -> bool:
    return all(map(lambda f: satisfies_filter(values, **f), filters))


def _any(values: dict, filters: list[dict]) -> bool:
    return any(map(lambda f: satisfies_filter(values, **f), filters))
