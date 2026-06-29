from __future__ import annotations

from typing import Protocol

from espresso_rl.domain.shadow_evaluation import DreamerShadowEvaluation


class ShadowEvaluationRepository(Protocol):
    def upsert(self, evaluation: DreamerShadowEvaluation) -> None:
        ...

    def get(self, evaluation_id: str) -> DreamerShadowEvaluation | None:
        ...

    def get_pending(
        self,
        *,
        install_id: str,
        machine_id: str,
        bean_context_id: str,
        grinder_context_id: str,
        inference_contract_id: str | None = None,
    ) -> DreamerShadowEvaluation | None:
        ...

    def list_context(
        self,
        *,
        install_id: str,
        machine_id: str,
        bean_context_id: str,
        grinder_context_id: str,
        inference_contract_id: str | None = None,
        limit: int = 100,
    ) -> list[DreamerShadowEvaluation]:
        ...
