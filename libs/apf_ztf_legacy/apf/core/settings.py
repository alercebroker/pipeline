import yaml


def config_from_yaml_file(path: str) -> dict:
    """Load a yaml file."""
    with open(path) as f:
        return yaml.load(f, Loader=yaml.SafeLoader)


def config_from_yaml_string(yaml_text: str):
    """Create a config object from a yaml text."""
    return yaml.load(yaml_text, Loader=yaml.SafeLoader)
