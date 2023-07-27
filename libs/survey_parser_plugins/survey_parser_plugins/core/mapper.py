from typing import Any, Callable, Dict, List, Union


class Mapper:
    """Class to generate mappings from alert messages to specific generic alert fields.

    Instances can be called with a message, which will execute the callable passed during construction
    with arguments given by fields in the message (defined by `origin` and `extras`).
    """

    def __init__(
        self,
        function: Callable = None,
        *,
        origin: str = None,
        extras: List[str] = None,
        required: bool = True
    ):
        """
        Arguments `function` and `origin` cannot be both `None`.

        If both `origin` and `extras` are `None`, the callable `function` expects no arguments. Otherwise,
        it expects a number of arguments equal to one (if `origin` is not `None`) plus the length of `extras`.

        The arguments passed to `function` are the value of the message field `origin` (if provided) followed
        by the values of the fields in `extras`.

        If `function` is not given, it will simply return the value of the field `origin`, ignoring `extras`.

        If the fields are not required, `None` will be used in their place.

        Args:
            function: A callable that generates the field, based on `origin` and `extras`
            origin: Main field from the alert message used to generate the generic alert field
            extras: Additional fields from the message that the function expects
            required: Whether it is required that the fields in `origin` and `extras` are present in the message
        """
        if origin is None and function is None:
            raise ValueError("Parameters 'origin' and 'function' cannot be both 'None'")
        self._origin = origin
        self._required = required
        self._function = function if function else lambda x: x
        self._extras = extras if extras else []

    @property
    def origin(self) -> Union[str, None]:
        """Main field in the message corresponding to the generic alert"""
        return self._origin

    def __call__(self, message: Dict[str, Any]) -> Any:
        """Execute the mapping based on the fields within message"""
        extras = [
            message[field] if self._required else message.get(field)
            for field in self._extras
        ]
        if self._origin is None:
            return self._function(*extras)
        origin = message[self._origin] if self._required else message.get(self._origin)
        return self._function(origin, *extras)
