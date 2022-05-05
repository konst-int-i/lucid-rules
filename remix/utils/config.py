import getpass
import os
import yaml
from typing import Dict, Union
from box import Box

class CustomYamlLoader(yaml.FullLoader):
    """Add a custom constructor "!include" to the YAML loader.
    "!include" allows to read parameters in another YAML file as if it was
    the main one.
    Examples:
        To read the parameters listed in credentials.yml and assign them to
        credentials in logging.yml:
        ``credentials: !include credentials.yml``
        To call: config.credentials
    """

    def __init__(self, stream):
        self._root = os.path.split(stream.name)[0]
        super(CustomYamlLoader, self).__init__(stream)

    def include(self, node: yaml.nodes.ScalarNode) -> Box:
        """Read yaml files as Box objects and overwrite user specific files
        Example: !include model.yml, will be overwritten by model.$USER.yml
        """

        filename: str = os.path.join(self._root, self.construct_scalar(node))
        subconfig: Box = _read(filename, loader=CustomYamlLoader)

        return subconfig


CustomYamlLoader.add_constructor("!include", CustomYamlLoader.include)

def _read(filename: str, loader) -> Box:
    """Read any yaml file as a Box object"""

    if os.path.isfile(filename) and os.access(filename, os.R_OK):
        with open(filename, "r") as f:
            try:
                config_dict = yaml.load(f, Loader=loader)
            except yaml.YAMLError as exc:
                print(exc)
        return Box(config_dict)
    else:
        raise FileNotFoundError(filename)


class Config:
    def __init__(self, config_path: str):
        self._config_path = config_path

    def read(self) -> Box:
        """Reads main config file"""
        if os.path.isfile(self._config_path) and os.access(self._config_path, os.R_OK):
            config = _read(filename=self._config_path, loader=CustomYamlLoader)
            return config
        else:
            raise FileNotFoundError(self._config_path)