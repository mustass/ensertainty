import os
from typing import List

from setuptools import setup

PROJ_DIR = os.path.dirname(__file__)


def load_requirements(
    file_name: str = "requirements.txt",
    dir_path: str = PROJ_DIR,
    comment_char: str = "#",
) -> List[str]:
    """Utility function to load requirements from a requirements.txt ignoring comments."""
    with open(os.path.join(dir_path, file_name)) as file:
        lines = [ln.strip() for ln in file.readlines()]
    reqs = []
    for ln in lines:
        if comment_char in ln:
            ln = ln[: ln.index(comment_char)].strip()
        if ln:
            reqs.append(ln)
    return reqs


setup(
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=load_requirements(),
    # extras_require={
    #    "dev": load_requirements("requirements-dev.txt"),
    # },
)
