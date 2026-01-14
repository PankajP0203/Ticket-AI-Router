from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    app_env: str = "dev"
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")

    chroma_persist_dir: str = "./.chroma"
    chroma_collection: str = "support_kb"

settings = Settings()
