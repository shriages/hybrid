from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class CallRecord:
    t: int
    model: str
    optimizer: str
    input_tokens: int
    output_tokens: int
    cost: float


class CostTracker:
    """
    Records every LLM API call and its exact cost.
    Produces the C_T cost curve used in all figures.
    """

    def __init__(self):
        self._records: List[CallRecord] = []

    def record(self, t: int, model: str, optimizer: str,
               input_tokens: int, output_tokens: int, cost: float) -> None:
        self._records.append(CallRecord(
            t=t, model=model, optimizer=optimizer,
            input_tokens=input_tokens, output_tokens=output_tokens, cost=cost,
        ))

    def total_cost(self) -> float:
        return sum(r.cost for r in self._records)

    def cost_at(self, t: int) -> float:
        """Cost incurred at exactly step t."""
        return sum(r.cost for r in self._records if r.t == t)  # so if it is multiple candidates' call, sum record all the costs

    def cost_curve(self, T: int) -> List[float]:
        """Cumulative cost curve, length T."""
        curve, total = [], 0.0
        for t in range(T):
            total += self.cost_at(t)
            curve.append(total)
        return curve

    def summary(self) -> Dict:
        return {
            "total_cost":    self.total_cost(),
            "total_calls":   len(self._records),
            "total_in_tok":  sum(r.input_tokens for r in self._records),
            "total_out_tok": sum(r.output_tokens for r in self._records),
        }