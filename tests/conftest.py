import pytest
from src.config import settings

# Force in-process PersistentClient mode for ChromaDB during test runs
settings.chroma_host = None

@pytest.fixture
def sample_fixture():
    return "fixture_value"
