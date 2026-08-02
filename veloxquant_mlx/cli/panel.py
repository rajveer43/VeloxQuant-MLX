"""``veloxquant panel`` — launch the local web control panel (#34)."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="veloxquant panel",
        description="Start the VeloxQuant control panel in your browser.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port for the panel itself (default: 7860). Always bound to "
        "127.0.0.1 — this API spawns processes and is never exposed.",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open a browser window.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    from veloxquant_mlx.ui.server import serve_panel

    serve_panel(port=args.port, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
