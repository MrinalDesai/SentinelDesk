from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Ollama
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    LLM_MODEL: str = "mistral:7b-instruct-q8_0"
    EMBEDDING_MODEL: str = "bge-m3"

    # Qdrant
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    COLLECTION_NAME: str = "sentineldesk_tickets"

    # PostgreSQL
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "sentineldesk"
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "sentineldesk123"

    # Agent settings
    CONFIDENCE_THRESHOLD: float = 0.75
    TOP_K_RESULTS: int = 5

    class Config:
        env_file = ".env"

settings = Settings()