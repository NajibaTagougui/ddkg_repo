import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pytest
torch = pytest.importorskip("torch")
from ddkg.selftest import VARIANTS, check_variant


@pytest.mark.parametrize("name", list(VARIANTS))
def test_variant(name):
    check_variant(name, VARIANTS[name], torch.device("cpu"))
