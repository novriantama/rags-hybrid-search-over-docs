import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # OpenAI Settings
    openai_api_key: str = ""
    llm_model: str = "gpt-4o"
    embedding_model: str = "text-embedding-3-small"

    # Database and Data Paths
    chroma_db_path: str = "./data/chroma"
    data_dir: str = "./data/documents"

    # API Settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Project Root Directory
    project_root: Path = Path(__file__).resolve().parent.parent

    # Configuration for Pydantic Settings loading from .env
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    def get_absolute_path(self, path_str: str) -> Path:
        """Resolve database and data paths relative to the project root if they are relative."""
        path = Path(path_str)
        if path.is_absolute():
            return path
        return (self.project_root / path).resolve()

settings = Settings()
