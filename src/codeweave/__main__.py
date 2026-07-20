from __future__ import annotations

import argparse


__version__ = "0.1.0"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="codeweave",
        description="CodeWeave Agent（码织 Agent）终端 AI 编程助手",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"CodeWeave Agent {__version__}",
    )
    return parser


def main() -> int:
    parser = build_parser()
    parser.parse_args()
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())