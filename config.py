# config.py
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()


class TokenTracker:

  def __init__(self):
    self.input_tokens = 0
    self.output_tokens = 0

  def add_usage(self, usage_metadata: dict):
    if usage_metadata:
      self.input_tokens += usage_metadata.get("input_tokens", 0)
      self.output_tokens += usage_metadata.get("output_tokens", 0)

  @property
  def total_tokens(self) -> int:
    return self.input_tokens + self.output_tokens

  def summary(self):
    print("--- Cumulative Token Summary ---")
    print(f"Input Tokens : {self.input_tokens}")
    print(f"Output Tokens: {self.output_tokens}")
    print(f"Total Tokens : {self.total_tokens}")


tracker = TokenTracker()


def get_llm(model_name: str = None, temperature: float = 0.0):
  """Factory for ChatGroq with dynamic model resolution and temperature."""
  if model_name is None:
    # Default to open-weight model on Groq or fall back to Llama 3.3
    model_name = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

  return ChatGroq(
      model=model_name,
      temperature=temperature,
      groq_api_key=os.environ.get("GROQ_API_KEY"),
  )