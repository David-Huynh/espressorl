from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Sequence

from espresso_rl.adapters.local_model_store import LocalModelArtifactStore
from espresso_rl.application.checkpoint_loading import DEFAULT_MAX_CHECKPOINT_BYTES
from espresso_rl.application.model_release import DreamerModelReleaseError, release_dreamer_checkpoint
from espresso_rl.domain.model_release import DreamerReleaseAuthorization


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    candidate_artifact = Path(args.candidate_artifact).resolve()
    candidate_manifest = Path(args.candidate_manifest).resolve()
    try:
        authorization = DreamerReleaseAuthorization(
            candidate_artifact_sha256=args.candidate_artifact_sha256,
            candidate_manifest_sha256=args.candidate_manifest_sha256,
            released_by=args.released_by,
            release_version=args.release_version,
            released_at=args.released_at if args.released_at is not None else int(time.time()),
        )
        result = release_dreamer_checkpoint(
            LocalModelArtifactStore(),
            candidate_artifact_reference=str(candidate_artifact),
            candidate_manifest_reference=str(candidate_manifest),
            authorization=authorization,
            max_checkpoint_bytes=args.max_checkpoint_bytes,
        )
        _write_bundle(
            Path(args.output_dir),
            result.files,
            protected_paths={candidate_artifact, candidate_manifest},
            force=args.force,
        )
        print(json.dumps(result.to_dict(), sort_keys=True, indent=2))
        return 0
    except (OSError, ValueError, DreamerModelReleaseError) as exc:
        print(f"espresso-rl Dreamer release failed: {exc}", file=sys.stderr)
        return 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify and explicitly authorize one DreamerV3 release-candidate checkpoint.",
    )
    parser.add_argument("--candidate-artifact", required=True, help="Path to candidate dreamer_v3.safetensors")
    parser.add_argument("--candidate-manifest", required=True, help="Path to candidate dreamer_v3_manifest.json")
    parser.add_argument("--candidate-artifact-sha256", required=True, help="Expected candidate artifact SHA-256")
    parser.add_argument("--candidate-manifest-sha256", required=True, help="Expected candidate manifest SHA-256")
    parser.add_argument("--released-by", required=True, help="Maintainer or release automation identity")
    parser.add_argument("--release-version", required=True, help="Release tag or immutable version identifier")
    parser.add_argument("--released-at", type=int, default=None, help="Optional deterministic Unix release timestamp")
    parser.add_argument("--output-dir", required=True, help="New directory for the inference-ready release bundle")
    parser.add_argument(
        "--max-checkpoint-bytes",
        type=int,
        default=DEFAULT_MAX_CHECKPOINT_BYTES,
        help="Maximum accepted candidate and released checkpoint size",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing release output files")
    return parser


def _write_bundle(output_dir: Path, files, *, protected_paths: set[Path], force: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    root = output_dir.resolve()
    targets: list[tuple[Path, object]] = []
    for artifact in files:
        relative = Path(artifact.relative_path)
        if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
            raise DreamerModelReleaseError("release artifact relative path is unsafe")
        target = (root / relative).resolve()
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise DreamerModelReleaseError("release artifact path escapes output directory") from exc
        if target in protected_paths:
            raise DreamerModelReleaseError("release output must not overwrite its candidate inputs")
        if target.exists() and not force:
            raise DreamerModelReleaseError(f"{target} already exists; use --force to overwrite")
        targets.append((target, artifact))

    for target, artifact in targets:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(artifact.content)


if __name__ == "__main__":
    raise SystemExit(main())
