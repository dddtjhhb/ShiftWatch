"""Service settings, read from environment variables with conservative defaults."""
from dataclasses import dataclass
import os


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Settings:
    database_url: str
    # Backpressure: total non-terminal tasks the system accepts before new runs get 429.
    max_pending_tasks: int
    # Largest single run accepted (cases x conditions).
    max_tasks_per_run: int
    default_max_attempts: int
    # Global concurrency cap for providers without a provider_limits row.
    default_provider_concurrency: int
    lease_seconds: float
    retry_base_seconds: float
    retry_max_seconds: float
    # fair: among equal-priority runs, the run with fewest running tasks goes next.
    # fifo: oldest run first (kept for the fairness experiment baseline).
    scheduler: str

    @classmethod
    def from_env(cls) -> "Settings":
        scheduler = _env("SHIFTWATCH_SCHEDULER", "fair")
        if scheduler not in {"fair", "fifo"}:
            raise ValueError("SHIFTWATCH_SCHEDULER must be fair or fifo")
        return cls(
            database_url=_env(
                "SHIFTWATCH_DATABASE_URL", "postgresql://postgres@localhost/shiftwatch"
            ),
            max_pending_tasks=int(_env("SHIFTWATCH_MAX_PENDING_TASKS", "20000")),
            max_tasks_per_run=int(_env("SHIFTWATCH_MAX_TASKS_PER_RUN", "5000")),
            default_max_attempts=int(_env("SHIFTWATCH_MAX_ATTEMPTS", "3")),
            default_provider_concurrency=int(_env("SHIFTWATCH_PROVIDER_CONCURRENCY", "4")),
            lease_seconds=float(_env("SHIFTWATCH_LEASE_SECONDS", "30")),
            retry_base_seconds=float(_env("SHIFTWATCH_RETRY_BASE_SECONDS", "2")),
            retry_max_seconds=float(_env("SHIFTWATCH_RETRY_MAX_SECONDS", "60")),
            scheduler=scheduler,
        )
