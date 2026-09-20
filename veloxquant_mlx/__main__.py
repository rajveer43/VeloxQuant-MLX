"""Entry point for `python -m veloxquant_mlx <command>`."""

from __future__ import annotations

import sys


def main() -> None:
    """Dispatch to the subcommand named in ``sys.argv[1]`` and run it.

    Registered as both the ``veloxquant`` and ``mlx-kv-quant`` console
    scripts (see ``pyproject.toml``). Rewrites ``sys.argv`` to strip the
    subcommand name before delegating, so each subcommand's own argument
    parser sees a clean ``argv`` as if it were invoked directly. Exits with
    status 1 (via ``sys.exit``) if no subcommand or an unknown one is given.
    """
    if len(sys.argv) < 2:
        print(
            "Usage: veloxquant "
            "{precompute|benchmark|recommend|auto-config|methods|serve|profile|"
            "profile-hardware|estimate-memory|panel|worker}"
        )
        sys.exit(1)

    command = sys.argv[1]
    # Remove the subcommand so sub-parsers see argv correctly
    sys.argv = [f"veloxquant_mlx {command}"] + sys.argv[2:]

    if command == "precompute":
        from veloxquant_mlx.cli.precompute import main as _main

        _main()
    elif command == "benchmark":
        from veloxquant_mlx.cli.benchmark import main as _main

        _main()
    elif command == "recommend":
        from veloxquant_mlx.cli.recommend import main as _main

        _main()
    elif command == "auto-config":
        from veloxquant_mlx.cli.auto_config import main as _main

        _main()
    elif command == "methods":
        from veloxquant_mlx.cli.methods import main as _main

        _main()
    elif command == "serve":
        from veloxquant_mlx.cli.serve import main as _main

        _main()
    elif command == "profile":
        from veloxquant_mlx.cli.profile import main as _main

        _main()
    elif command == "profile-hardware":
        from veloxquant_mlx.cli.profile_hardware import main as _main

        _main()
    elif command == "estimate-memory":
        from veloxquant_mlx.cli.estimate_memory import main as _main

        _main()
    elif command == "panel":
        from veloxquant_mlx.cli.panel import main as _main

        _main()
    elif command == "worker":
        from veloxquant_mlx.cli.worker import main as _main

        _main()
    else:
        print(
            f"Unknown command: {command!r}. "
            "Choices: precompute, benchmark, recommend, auto-config, methods, "
            "serve, profile, profile-hardware, estimate-memory, panel, worker"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
