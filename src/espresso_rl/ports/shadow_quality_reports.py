from __future__ import annotations

from typing import Protocol

from espresso_rl.domain.shadow_quality import DreamerShadowQualityReport


class ShadowQualityReportRepository(Protocol):
    def upsert(self, report: DreamerShadowQualityReport) -> None:
        ...

    def get(self, report_id: str) -> DreamerShadowQualityReport | None:
        ...

    def get_latest(
        self,
        *,
        install_id: str,
        machine_id: str,
        bean_context_id: str,
        grinder_context_id: str,
        checkpoint_artifact_sha256: str,
        checkpoint_inference_probe_sha256: str,
        inference_contract_id: str,
    ) -> DreamerShadowQualityReport | None:
        ...
