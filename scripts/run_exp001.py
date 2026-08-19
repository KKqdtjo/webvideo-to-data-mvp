"""Command-line entry point for one EXP-001 variant."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from webvideo_to_data.experiment import run_experiment


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--variant", choices=("B0", "B1", "B2", "B3", "B4"), required=True
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--no-render", action="store_true")
    arguments = parser.parse_args()
    output_dir = arguments.output_dir or Path("artifacts") / "EXP-001" / arguments.variant
    metrics = run_experiment(
        arguments.config,
        output_dir,
        variant=arguments.variant,
        no_render=arguments.no_render,
    )
    print(json.dumps(metrics, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
