from abc import ABC, abstractmethod


class BaseLLM(ABC):
    """Abstract LLM backend. Each subclass wraps one API provider."""

    model: str

    @abstractmethod
    def complete(self, prompt: str, max_tokens: int = 256) -> dict:
        """
        Send prompt and return a response dict:
        {
            "text":         str,   # the model's reply
            "input_tokens": int,   # tokens consumed in prompt
            "output_tokens": int,  # tokens in completion
            "cost":         float, # dollar cost of this call
        }
        """
        ...

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using this model's tokenizer."""
        ...