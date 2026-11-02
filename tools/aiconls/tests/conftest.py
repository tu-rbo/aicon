import sys
from pathlib import Path

import pytest

TESTS_ROOT = Path(__file__).resolve().parent
REPO_ROOT = TESTS_ROOT.parent
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return TESTS_ROOT / "fixtures"


@pytest.fixture(scope="session")
def sample_spec_path(fixtures_dir: Path) -> Path:
    return fixtures_dir / "sample_spec.yaml"


@pytest.fixture(scope="session")
def sample_spec_content(sample_spec_path: Path) -> str:
    return sample_spec_path.read_text()


@pytest.fixture(scope="session")
def complex_spec_path(fixtures_dir: Path) -> Path:
    return fixtures_dir / "complex_spec.yaml"


@pytest.fixture(scope="session")
def complex_spec_content(complex_spec_path: Path) -> str:
    return complex_spec_path.read_text()
