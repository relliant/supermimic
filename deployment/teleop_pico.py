"""Command-line entry point; implementation: deployment.teleop.pico."""

if __name__ == "__main__":
    if not __package__:
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from deployment.teleop.pico import main

    raise SystemExit(main())
