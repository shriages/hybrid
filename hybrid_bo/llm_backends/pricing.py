"""
API pricing table. All prices in USD per 1,000,000 tokens (standard quoting unit).
To get cost for N tokens: cost = (N / 1_000_000) * price_per_million
"""

MODEL_PRICING = {
    # OpenAI
    "gpt-3.5-turbo":       {"input": 0.50,   "output": 1.50},
    "gpt-4-turbo":         {"input": 10.00,  "output": 30.00},
    "gpt-4o":              {"input": 5.00,   "output": 15.00},
    "gpt-4o-mini":         {"input": 0.15,   "output": 0.60},
    "gpt-5.4-mini":        {"input": 0.75,   "output": 4.50},
    # Anthropic
    "claude-3-5-sonnet":   {"input": 3.00,   "output": 15.00},
    "claude-3-haiku":      {"input": 0.25,   "output": 1.25},
    # DeepSeek
    "deepseek-v4-flash":   {"input": 0.14,   "output": 0.28},
    # Local (Llama) — no API cost
    "llama-local":         {"input": 0.00,   "output": 0.00},
    # Anthropic
    "claude-3-5-sonnet":   {"input": 3.00,   "output": 15.00},
    "claude-3-haiku":      {"input": 0.25,   "output": 1.25},
    "claude-sonnet-5":     {"input": 2.00,   "output": 10.00},   # <-- add
    "claude-haiku-4-5":    {"input": 1.00,   "output": 5.00},
}


def compute_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Return dollar cost for one API call given token counts."""
    if model not in MODEL_PRICING:
        raise ValueError(f"Unknown model '{model}'. Add it to MODEL_PRICING.")
    p = MODEL_PRICING[model]
    return (input_tokens * p["input"] + output_tokens * p["output"]) / 1_000_000