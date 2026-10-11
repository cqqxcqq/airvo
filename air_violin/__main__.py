import sys

try:
    from .app import main
except ModuleNotFoundError as exc:
    print(f"Missing dependency: {exc.name}. Activate your venv and run python -m pip install -r requirements.txt", file=sys.stderr)
    raise SystemExit(2)

raise SystemExit(main())
