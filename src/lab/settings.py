from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings, loaded from environment variables or .env file in the project root."""

    # LiteLLM API settings. 
    litellm_base_url: str = "https://gpustack.ing.unibs.it/v1"
    litellm_api_key: str | None = None # Remember to set this in your .env file or environment variables!

    # Model settings. These are mostly the defaults recommended here: https://huggingface.co/Qwen/Qwen3.5-9B#best-practices
    model: str = "qwen35-4b"
    max_tokens: int = 32768
    temperature: float = 0 # For a more deterministic output
    top_p: float = 0.8
    top_k: int = 20
    presence_penalty: float = 2.0
    repetition_penalty: float = 1.0

    # External endpoints
    wikidata_endpoint: str = "https://query.wikidata.org/sparql"
    wikidata_user_agent: str = "UniBS-Lab/1.0 (university lab; non-commercial)"

    @property
    def litellm_model(self) -> str:
        """Return a LiteLLM-compatible model name."""

        return self.model if "/" in self.model else f"openai/{self.model}"
