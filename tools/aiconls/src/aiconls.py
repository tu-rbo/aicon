import argparse
import re
import sys
from collections.abc import Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from project_scanner import (
    ScanReport,
    apply_missing_quantity_definitions,
    format_report,
    format_stylish_report,
    scan_project,
)


def resolve_cli_version() -> str:
    try:
        return version("AICONLS")
    except PackageNotFoundError:
        pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
        content = pyproject_path.read_text(encoding="utf-8")
        match = re.search(r'^version = "([^"]+)"$', content, flags=re.MULTILINE)
        if match is None:
            raise RuntimeError(
                "Could not determine AICONLS version from pyproject.toml"
            )
        return match.group(1)


def start_language_server(server_args: list[str]) -> None:
    # imported lazily so `aiconls scan` does not pay for the server setup
    from pygls.cli import start_server

    from handlers import register_handlers
    from lsp_utils import configure_debugpy_from_env, configure_logging
    from server import create_server

    server = create_server()
    register_handlers(server)
    configure_logging()
    configure_debugpy_from_env()
    start_server(server, server_args)


def _resolve_exit_code(report: ScanReport, fail_on: str) -> int:
    summary = report["summary"]
    diagnostic_count = summary["diagnostic_count"]
    error_count = summary["error_count"]

    if fail_on == "none":
        return 0
    if fail_on == "error":
        return 1 if error_count > 0 else 0
    return 1 if diagnostic_count > 0 else 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="aiconls",
        description="AICON language server utilities.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {resolve_cli_version()}",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    lsp_parser = subparsers.add_parser(
        "lsp",
        help="Start the language server over stdio (project root: AICONLS_PROJECT_ROOT).",
    )
    lsp_parser.add_argument(
        "server_args",
        nargs=argparse.REMAINDER,
        help="Transport options passed to pygls, e.g. --tcp --port 8888.",
    )

    scan_parser = subparsers.add_parser(
        "scan",
        help="Scan an AICON project and emit structured diagnostics.",
    )
    scan_parser.add_argument(
        "project",
        nargs="?",
        default=".",
        help="Path to the project root (defaults to current directory).",
    )
    scan_parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output.",
    )
    scan_parser.add_argument(
        "--format",
        choices=("stylish", "json"),
        default="stylish",
        help="Output format. Use 'stylish' for human-readable output.",
    )
    scan_parser.add_argument(
        "--output",
        type=Path,
        help="Write output to a file instead of stdout.",
    )
    scan_parser.add_argument(
        "--fail-on",
        choices=("none", "error", "warning"),
        default="warning",
        help="Set when to return a non-zero exit code.",
    )
    scan_parser.add_argument(
        "--color",
        choices=("auto", "always", "never"),
        default="auto",
        help="Enable ANSI colors for stylish output.",
    )

    scan_parser.add_argument(
        "--fix",
        action="store_true",
        help="Insert missing cls_quantity_names_and_shapes definitions from spec.",
    )
    scan_parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Use dynamic parse.",
    )

    args = parser.parse_args(argv)

    if args.command == "lsp":
        start_language_server(args.server_args)
        return 0

    if args.command == "scan":
        if args.fix:
            apply_missing_quantity_definitions(Path(args.project))
        report = scan_project(Path(args.project))
        report = scan_project(Path(args.project), args.dynamic)
        if args.format == "json":
            output = format_report(report, pretty=args.pretty)
        else:
            use_color = False
            if args.color == "always":
                use_color = True
            elif args.color == "auto":
                use_color = args.output is None and sys.stdout.isatty()
            output = format_stylish_report(report, use_color=use_color)
        if args.output is None:
            print(output)
        else:
            args.output.write_text(f"{output}\n", encoding="utf-8")
        return _resolve_exit_code(report, args.fail_on)

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
