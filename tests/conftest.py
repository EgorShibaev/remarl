import pytest
from remarl.baselines.sources import source_root


@pytest.fixture
def upstream_sources():
    root = source_root()
    if not (root / "WINO/LLaDA/modeling_llada.py").exists():
        pytest.skip(
            "Run remarl download baseline-sources --output assets/baselines for reference tests"
        )
    return root
