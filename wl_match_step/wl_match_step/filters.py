def constant(values: dict, field: str, constant: int, op: str):
    value = values[field]
    match op:
        case "less":
            return value < constant
        case "less eq":
            return value <= constant
        case "greater":
            return value > constant
        case "greater eq":
            return value >= constant
        case _:
            raise Exception("invalid value for filter.params.op")
