from typing import Any, Callable, Dict, Union


class Mapper:
    def __init__(self, field: str, function: Callable[[Dict[str, Any]], Any] = None, *, origin: str = None):
        if origin is None and function is None:
            raise ValueError("Parameters 'origin' and 'function' cannot be both 'None'")
        self._field = field
        self._origin = origin
        self._function = function

    @property
    def field(self) -> str:
        return self._field

    @property
    def origin(self) -> Union[str, None]:
        return self._origin

    def __call__(self, message: Dict[str, Any]):
        if self._function is not None and self._origin is not None:
            return self._function(message, self._origin)
        if self._function is not None and self._origin is None:
            return self._function(message)
        return message[self.origin]
