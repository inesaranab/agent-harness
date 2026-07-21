from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")
    openai_api_key: str
    database_url: str
    e2b_api_key: str
    rpc_base_url: str

    @field_validator("rpc_base_url", "database_url", "e2b_api_key", "openai_api_key")
    @classmethod
    def _strip(cls, v: str) -> str:
        # A stray space in .env (e.g. "RPC_BASE_URL= https://…") silently breaks
        # the URL, so trim whitespace on the way in.
        return v.strip()


settings = Settings()
