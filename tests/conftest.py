import pytest

from klia.config import load_config
from klia.demo import make


@pytest.fixture(scope="session")
def cfg():
    c = load_config()
    c["model"] = {**c["model"], "name": "sgd_log", "params": {}, "warmup_rows": 200}
    # Disable the promotion gate for unit tests.  The gate is an optional production
    # feature; when it is enabled in config.yaml the incremental tests that assert on
    # n_learned will fail because a rejected candidate leaves the incumbent (bootstrap
    # bundle) committed instead of the newly-trained one.
    c["gate"] = {**c.get("gate", {}), "enabled": False}
    return c


@pytest.fixture(scope="session")
def demo_df():
    return make(rows=3000, seed=3)


@pytest.fixture()
def tmp_artifacts(tmp_path, monkeypatch):
    monkeypatch.setenv("KLIA_ARTIFACTS_DIR", str(tmp_path / "art"))
    monkeypatch.delenv("DATABASE_URL", raising=False)
    return tmp_path