import os
import pathlib

from _pytest.config import Config


def pytest_configure(config: Config) -> None:  # noqa: ARG001
    root_dir = pathlib.Path(__file__).parent.parent.resolve()
    os.chdir(str(root_dir))
