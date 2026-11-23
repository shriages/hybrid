import os
import time
import random
import openai
from .base_llm import BaseLLM
from .pricing import compute_cost


def _is_reasoning_model(model: str) -> bool:
    """GPT-5.x, o1, o3, o4 are reasoning models with a different API surface:
    they use max_completion_tokens (not max_tokens), reject temperature!=1,
    and emit billed reasoning tokens."""
    m = model.lower()
    return m.startswith(("gpt-5", "o1", "o3", "o4"))


class OpenAILLM(BaseLLM):
    """OpenAI API backend.

    Handles BOTH the classic chat models (gpt-3.5/4/4o: max_tokens +
    temperature) and the newer reasoning models (gpt-5.x/o-series:
    max_completion_tokens, no custom temperature, reasoning_effort).
    """

    def __init__(self, model: str = "gpt-3.5-turbo", max_retries: int = 20,
                 reasoning_effort: str = None):
        import tiktoken
        self.model = model
        self.max_retries = max_retries
        self.reasoning = _is_reasoning_model(model)
        # reasoning_effort: None means "don't send it" (plain max_completion_tokens,
        # which is the verified-working config for gpt-5.4-mini). If set, must be
        # one of the model's supported values: none/low/medium/high/xhigh
        # (NOT 'minimal' — gpt-5.4-mini rejects that).
        self.reasoning_effort = reasoning_effort
        self._client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        try:
            self._enc = tiktoken.encoding_for_model(model)
        except Exception:
            self._enc = tiktoken.get_encoding("o200k_base")

    def count_tokens(self, text: str) -> int:
        return len(self._enc.encode(text))

    def _wait(self, attempt: int, base: float = 8.0) -> None:
        cap = 64.0
        slot = min(base * (2 ** attempt), cap)
        wait = random.uniform(slot * 0.5, slot)
        print(f"[OpenAI] backoff {wait:.1f}s (attempt {attempt+1}/{self.max_retries})")
        time.sleep(wait)

    def _build_kwargs(self, prompt: str, max_tokens: int) -> dict:
        kwargs = dict(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        if self.reasoning:
            # Reasoning models: token budget must cover hidden reasoning tokens,
            # so give generous headroom; temperature is NOT set (must default).
            kwargs["max_completion_tokens"] = max(max_tokens, 2048)
            # Only pass reasoning_effort if explicitly configured. gpt-5.4-mini
            # works fine WITHOUT it (verified), and rejects 'minimal'. Valid
            # values if you do set it: none/low/medium/high/xhigh.
            if self.reasoning_effort is not None:
                kwargs["reasoning_effort"] = self.reasoning_effort
        else:
            kwargs["max_tokens"] = max_tokens
            kwargs["temperature"] = 0.0
        return kwargs

    def complete(self, prompt: str, max_tokens: int = 256, json_only=True) -> dict:
        for attempt in range(self.max_retries):
            try:
                kwargs = self._build_kwargs(prompt, max_tokens)
                response = self._client.chat.completions.create(**kwargs)

                usage = response.usage
                in_tok = usage.prompt_tokens
                out_tok = usage.completion_tokens

                text = response.choices[0].message.content or ""
                return {
                    "text": text,
                    "input_tokens": in_tok,
                    "output_tokens": out_tok,
                    "cost": compute_cost(self.model, in_tok, out_tok),
                }

            except openai.AuthenticationError as e:
                raise RuntimeError(f"OpenAI authentication failed: {e}") from e

            except openai.BadRequestError as e:
                # A parameter the model rejects (e.g. temperature, max_tokens).
                # Surface it loudly — retrying won't help a 400.
                print(f"[OpenAI] BadRequest (won't retry): {str(e)[:200]}")
                raise RuntimeError(f"OpenAI rejected the request: {e}") from e

            except openai.RateLimitError as e:
                print("STATUS:", getattr(e, "status_code", "?"),
                      "TYPE:", getattr(e, "type", None))
                self._wait(attempt, base=10.0)

            except openai.InternalServerError:
                self._wait(attempt, base=5.0)

            except openai.APIConnectionError as e:
                print(f"[OpenAI] Connection error: {str(e)[:120]}")
                self._wait(attempt, base=5.0)

            except openai.APIError as e:
                print(f"[OpenAI] APIError {type(e).__name__}: {str(e)[:120]}")
                self._wait(attempt, base=5.0)

        raise RuntimeError(
            f"OpenAI API failed after {self.max_retries} retries "
            f"(model={self.model})."
        )