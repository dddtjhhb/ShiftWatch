# Service integration status and follow-up issues

## Scope and provenance

Integrated the user-supplied `shiftwatch-service-platform.patch` into the local repository. Preserved the pre-existing `score_llm_response` refactor and the untracked `shiftwatch/platform/` implementation. Added the missing `case_from_dict` interface required by the service. The service entry point is `shiftwatch.service`; the older platform draft has not been removed or merged.

The supplied `service_experiments.json` and numerical tables are retained for provenance. Their execution environment and results have not been verified here. Do not cite their numbers as reproduced performance evidence.

## Validation

- `python3 -m unittest discover -q`: 45 discovered, 23 passed, 22 service integration tests skipped.
- Dependency installation blocked by unavailable PyPI DNS/network access.
- Docker daemon access denied in this execution environment; PostgreSQL tests and crash experiments remain pending.
- No remote commit or push performed.

## Findings to resolve before claiming reliability guarantees

1. **Expired lease can be revived or accepted before reaping.** `heartbeat`, `complete_task`, and `fail_task` check token/status but not lease expiry. Reproduce by expiring a lease directly, then heartbeating/completing before calling the reaper. Add database-clock expiry predicates and integration coverage.
2. **Cancelled runs can retain unschedulable retry tasks.** Cancelling leaves in-flight work running. A later retryable failure or expired lease moves that task to `retry_wait`, while claimers exclude cancelled runs. Ensure cancellation wins over retry/reaping and that the run reaches a terminal state. Test cancel plus provider failure and cancel plus worker crash.
3. **Provider concurrency limit covers database leases, not all live remote calls.** If an expired attempt still executes remotely, reassigning its slot can overlap external calls. Report this boundary explicitly and test slow requests plus lost heartbeats; do not promise an absolute remote concurrency cap.
4. **Candidate window may starve another provider.** `_candidate_runs` limits candidates to 25 before checking provider capacity. Twenty-five runs for a saturated provider can hide runnable work for a different provider. Add a test with more than 25 runs and revisit provider selection.
5. **Cancellation and completion use opposite lock order.** Cancellation locks a run before its tasks; completion locks a task before its run. Review concurrent cancellation/completion for PostgreSQL deadlocks and establish a consistent lock order or bounded transaction retry.
6. **Fairness is a heuristic, not a starvation guarantee.** Fewest-running-first can repeatedly choose an older run at capacity one; candidate ordering can also become stale under concurrent claims. Test capacity one, multiple providers and sustained submissions before claiming bounded waiting.

## Acceptance steps

Run PostgreSQL integration tests in a dedicated disposable database (the suite truncates its service tables), address the findings above with regression cases, then reproduce fault/scaling experiments and record fresh environment metadata. Finally run one real Ollama end-to-end evaluation and rescore its saved outputs.
