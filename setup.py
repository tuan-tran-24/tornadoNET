from pathlib import Path
from setuptools import setup

PROJECT_ROOT = Path(__file__).parent.resolve()
REQUIREMENTS_FILE = PROJECT_ROOT / "requirements.txt"


def is_valid_module_name(name: str) -> bool:
    return name.isidentifier()


def parse_requirements(path: Path) -> list[str]:
    """Read install requirements from requirements.txt."""
    requirements: list[str] = []
    if not path.exists():
        return requirements

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(("-r", "--requirement", "-c", "--constraint")):
            continue
        requirements.append(line)
    return requirements


# Collect top-level .py modules in this folder.
# Files with names that are not valid Python identifiers are skipped.
py_modules = sorted(
    p.stem
    for p in PROJECT_ROOT.glob("*.py")
    if p.name != "setup.py" and is_valid_module_name(p.stem)
)

install_requires = parse_requirements(REQUIREMENTS_FILE)

setup(
    name="recovery-modeling-utils",
    version="0.1.0",
    description="Utilities and training code for single-stream and hybrid recovery assessment models.",
    author="",
    py_modules=py_modules,
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=install_requires,
)
