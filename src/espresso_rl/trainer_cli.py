from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Sequence

from espresso_rl.application.trainer_artifacts import (
    DEFAULT_MAX_DATASET_BYTES,
    TrainerArtifactError,
    build_dreamer_trainer_artifacts,
)
from espresso_rl.domain.trainer_artifacts import default_training_config


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.write_default_config:
            _write_default_config(
                Path(args.write_default_config),
                seed=args.seed,
                artifact_stage=args.artifact_stage,
                force=args.force,
            )
            return 0
        _require_build_args(args)
        dataset_path = Path(args.dataset_jsonl)
        _enforce_input_file_size(dataset_path, max_bytes=args.max_dataset_bytes)
        result = build_dreamer_trainer_artifacts(
            training_rows_jsonl=dataset_path.read_text(encoding="utf-8"),
            training_dataset_manifest_json=Path(args.dataset_manifest).read_text(encoding="utf-8"),
            training_config_json=Path(args.training_config).read_text(encoding="utf-8"),
            trainer_git_sha=args.trainer_git_sha,
            created_at=args.created_at if args.created_at is not None else int(time.time()),
            max_dataset_bytes=args.max_dataset_bytes,
        )
        output_dir = Path(args.output_dir)
        _write_bundle(output_dir, result.files, force=args.force)
        print(json.dumps(result.to_dict(), sort_keys=True, indent=2))
        return 0
    except (OSError, TrainerArtifactError, ValueError) as exc:
        print(f"espresso-rl trainer artifact build failed: {exc}", file=sys.stderr)
        return 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build auditable DreamerV3 artifact-contract files from a canonical EspressoRL training export.",
    )
    parser.add_argument("--dataset-jsonl", help="Path to training_rows.jsonl")
    parser.add_argument("--dataset-manifest", help="Path to the training export manifest.json")
    parser.add_argument("--training-config", help="Path to training_config.json")
    parser.add_argument("--output-dir", help="Directory where artifact files will be written")
    parser.add_argument("--trainer-git-sha", help="Trainer repository commit SHA or build identifier")
    parser.add_argument("--created-at", type=int, default=None, help="Optional deterministic created_at timestamp")
    parser.add_argument(
        "--max-dataset-bytes",
        type=int,
        default=DEFAULT_MAX_DATASET_BYTES,
        help=(
            "Local resource-safety limit for this artifact skeleton command. "
            "Defaults to 8 GiB; real trainers should use streaming/sharded loading for larger corpora."
        ),
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing output files")
    parser.add_argument(
        "--write-default-config",
        help="Write a minimal artifact-contract-only training_config.json and exit",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed used by --write-default-config")
    parser.add_argument(
        "--artifact-stage",
        default="artifact_contract_only",
        choices=(
            "artifact_contract_only",
            "world_model_smoke",
            "world_model_train_preview",
            "world_model_release_candidate",
        ),
        help="Artifact stage used by --write-default-config",
    )
    return parser


def _require_build_args(args: argparse.Namespace) -> None:
    missing = [
        name
        for name in ("dataset_jsonl", "dataset_manifest", "training_config", "output_dir", "trainer_git_sha")
        if not getattr(args, name)
    ]
    if missing:
        raise TrainerArtifactError(f"missing required arguments: {', '.join('--' + name.replace('_', '-') for name in missing)}")


def _write_default_config(path: Path, *, seed: int, artifact_stage: str, force: bool) -> None:
    if path.exists() and not force:
        raise TrainerArtifactError(f"{path} already exists; use --force to overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(default_training_config(seed=seed, artifact_stage=artifact_stage), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _enforce_input_file_size(path: Path, *, max_bytes: int) -> None:
    if max_bytes <= 0:
        raise TrainerArtifactError("--max-dataset-bytes must be positive")
    size = path.stat().st_size
    if size <= 0:
        raise TrainerArtifactError(f"{path} is empty")
    if size > max_bytes:
        raise TrainerArtifactError(f"{path} is larger than --max-dataset-bytes")


def _write_bundle(output_dir: Path, files, *, force: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    root = output_dir.resolve()
    targets: list[tuple[Path, object]] = []
    for artifact in files:
        relative = Path(artifact.relative_path)
        if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
            raise TrainerArtifactError("artifact relative_path is unsafe")
        target = (root / relative).resolve()
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise TrainerArtifactError("artifact output path escapes output directory") from exc
        if target.exists() and not force:
            raise TrainerArtifactError(f"{target} already exists; use --force to overwrite")
        targets.append((target, artifact))

    for target, artifact in targets:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(artifact.content)


if __name__ == "__main__":
    raise SystemExit(main())
