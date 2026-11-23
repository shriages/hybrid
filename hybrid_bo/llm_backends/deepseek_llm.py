import os
from .base_llm import BaseLLM
from .pricing import compute_cost


class DeepSeekLLM(BaseLLM):
    """DeepSeek API backend (OpenAI-compatible)."""

    def __init__(self, model: str = "deepseek-v4-flash", thinking: bool = False):
        import openai
        import tiktoken
        self.model = model
        self.thinking = thinking
        self._client = openai.OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url="https://api.deepseek.com",
        )
        # DeepSeek tokenization is close to GPT; cl100k is an approximation.
        self._enc = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        return len(self._enc.encode(text))


    def complete(self, prompt: str, max_tokens: int = 256,
                    temperature: float = 0.7, json_only: bool = True) -> dict:
            kwargs = dict(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
            )

            if json_only:
                # Guarantees syntactically valid JSON (object), suppressing prose.
                kwargs["response_format"] = {"type": "json_object"}

            if self.thinking:
                kwargs["reasoning_effort"] = "high"
                kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
            else:
                kwargs["temperature"] = temperature
                kwargs["extra_body"] = {"thinking": {"type": "disabled"}}

            response = self._client.chat.completions.create(**kwargs)

            msg = response.choices[0].message
            text = msg.content or ""

            in_tok = response.usage.prompt_tokens
            out_tok = response.usage.completion_tokens
            return {
                "text":          text,
                "input_tokens":  in_tok,
                "output_tokens": out_tok,
                "cost":          compute_cost(self.model, in_tok, out_tok),
            }