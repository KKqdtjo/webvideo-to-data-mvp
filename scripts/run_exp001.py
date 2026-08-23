"""Command-line entry point for one EXP-001 variant."""

from __future__ import annotations

import argparse
from pathlib import Path

from webvideo_to_data.cli import main as cli_main


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
    argv = [
        "run",
        "--config",
        str(arguments.config),
        "--variant",
        arguments.variant,
        "--output-dir",
        str(output_dir),
        "--json",
    ]
    if arguments.no_render:
        argv.append("--no-render")
    return cli_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
