from typing import Any, Callable, Dict, Union


class Mapper:
    def __init__(self, field: str, function: Callable = None, *, origin: str = None, extras: list[str] = None, required: bool = True):
        if origin is None and function is None:
            raise ValueError("Parameters 'origin' and 'function' cannot be both 'None'")
        self._field = field
        self._origin = origin
        self._required = required
        self._function = function if function else lambda x: x
        self._extras = extras if extras else []

    @property
    def field(self) -> str:
        return self._field

    @property
    def origin(self) -> Union[str, None]:
        return self._origin

    def __call__(self, message: Dict[str, Any]):
        extras = [message[field] if self._required else message.get(field) for field in self._extras]
        if self._origin is None:
            return self._function(*extras)
        origin = message[self._origin] if self._required else message.get(self._origin)
        return self._function(origin, *extras)
