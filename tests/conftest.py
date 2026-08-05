import pytest

# agent.py calls load_dotenv() at import time, so whatever sits in the developer's .env
# lands in os.environ for the whole session. Wipe our own vars before each test, otherwise
# a stray FORCE_CONFIDENCE in someone's .env shows up as a mystery failure in two tests
# that have nothing to do with their change.
_OUR_VARS = (
    "HIGH_CONFIDENCE",
    "LOW_CONFIDENCE",
    "LOW_CONFIDENCE_STRATEGY",
    "CLARIFICATION_PROMPT",
    "FORCE_CONFIDENCE",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _OUR_VARS:
        monkeypatch.delenv(name, raising=False)
