

class BaseParser:
    """
    Docstring for BaseParser
    Base class for schema parsers. It only works with dictionaries.
    """

    def __init__(self, source_data: dict):
        self.source_data = source_data

    def get_parse_map(self):
        """
        Returns a mapping of fields to be parsed.
        Subclasses should implement this method.
        """
        raise NotImplementedError("Subclasses should implement this method")
    
    def parse(self, data):
        result_dict = {}
        
        for key, value in data.items():
            if callable(value):
                result_dict[key] = value()
            else:
                result_dict[key] = value
    
    def copy_field(self, source_data, field_name, rename_fields: dict[str, str] | None = None):
        """
        Copy a field from source_data to a new dictionary, optionally renaming some fields in it.
        """
        def _copy_field():
            source_field = source_data.get(field_name)

            if rename_fields:
                for key in source_field.keys():
                    if key in rename_fields:
                        source_field[rename_fields[key]] = source_field.pop(key)
        
        return _copy_field

        