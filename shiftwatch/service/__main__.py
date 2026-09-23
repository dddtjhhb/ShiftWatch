"""Entry point: python -m shiftwatch.service {migrate,api,worker,mock-provider,client} ..."""
import sys


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print(__doc__)
        return
    command, rest = sys.argv[1], sys.argv[2:]
    if command == "migrate":
        from .config import Settings
        from .db import migrate

        applied = migrate(Settings.from_env().database_url)
        print(f"applied: {applied or 'nothing (up to date)'}")
    elif command == "api":
        import uvicorn

        host = rest[0] if rest else "0.0.0.0"
        uvicorn.run("shiftwatch.service.api:create_app", factory=True, host=host, port=8000)
    elif command == "worker":
        from .worker import main as worker_main

        worker_main(rest)
    elif command == "mock-provider":
        from .mock_provider import main as mock_main

        mock_main(rest)
    elif command == "client":
        from .client import main as client_main

        client_main(rest)
    else:
        sys.exit(f"unknown command {command!r}")


if __name__ == "__main__":
    main()
