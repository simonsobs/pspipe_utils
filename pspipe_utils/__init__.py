import os


def get_data_path():
    return os.path.join(os.path.dirname(__file__), "data")

from . import _version
__version__ = _version.get_versions()['version']
