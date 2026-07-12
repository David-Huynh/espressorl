from __future__ import annotations

import argparse
import sys
from pathlib import Path

from espresso_rl.adapters.file_offline_dataset import write_offline_dataset_export
from espresso_rl.adapters.postgres_repositories import PostgresCommunityWarehouse, PostgresStore
from espresso_rl.application.offline_dataset_export import OfflineDatasetExportService
from espresso_rl.config import Config


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    try:
        config = Config.load()
        dsn = str(args.postgres_dsn or config.postgres_dsn).strip()
        if not dsn:
            raise ValueError("Postgres DSN is required")
        warehouse = PostgresCommunityWarehouse(PostgresStore(dsn))
        export = OfflineDatasetExportService(
            warehouse,
            clock=config.now,
            exporter_version=config.build_git_sha or "development",
        ).build(limit=args.limit)
        paths = write_offline_dataset_export(
            export,
            Path(args.output_dir),
            force=args.force,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"offline dataset export failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    print(f"exported {export.manifest['record_count']} comparisons")
    for path in paths:
        print(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export validated physical shot trajectories and pairwise preferences "
            "as plain JSONL plus a SHA-256 manifest."
        )
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--postgres-dsn")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    return parser


if __name__ == "__main__":
    main()
