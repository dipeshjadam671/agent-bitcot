import os
from pathlib import Path
from dotenv import load_dotenv

# Base directory for the repository
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from .env file
load_dotenv(BASE_DIR / ".env")


class Settings:
    def __init__(self):
        self.BASE_DIR: Path = BASE_DIR
        self.OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
        self.OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.OPENAI_EMBEDDING_MODEL: str = os.getenv(
            "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
        )
        self.TRIPMATE_LOG_LEVEL: str = os.getenv("TRIPMATE_LOG_LEVEL", "INFO").upper()
        self.TRIPMATE_VERBOSE: bool = os.getenv("TRIPMATE_VERBOSE", "true").lower() in (
            "1",
            "true",
            "yes",
        )
        self.LOG_FILE: Path = BASE_DIR / os.getenv("LOG_FILE", "logs/tripmate.log")
        self.DATA_DIR: Path = BASE_DIR / os.getenv("DATA_DIR", "data")
        self.RAW_DATA_DIR: Path = self.DATA_DIR / "destination_guide" / "raw"
        self.TOP_K_CHUNKS: int = int(os.getenv("TOP_K_CHUNKS", "3"))
        self.OPEN_METEO_TIMEOUT_SECONDS: int = int(
            os.getenv("OPEN_METEO_TIMEOUT_SECONDS", "5")
        )
        self.SUPPORTED_CITIES: list[str] = [
            "Bangkok",
            "Barcelona",
            "Reykjavik",
            "Tokyo",
        ]

    def validate_openai_key(self) -> bool:
        """Returns True if a plausible OpenAI API key is configured."""
        return bool(
            self.OPENAI_API_KEY
            and not self.OPENAI_API_KEY.startswith("mock-")
            and not self.OPENAI_API_KEY.startswith("your_")
        )


settings = Settings()
