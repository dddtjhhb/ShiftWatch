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

## Resolution of findings (follow-up patch)

Each fix has a regression test in `tests/test_service.py`. The two bugs (2 and 5) and the scheduling gaps (4 and 6) were first reproduced as failing tests against the previous code.

1. **Expired lease before reaping. Kept by design and now documented.** The lease token is the fencing mechanism. Until the reaper moves the task on, no other attempt can hold it, so the original holder's result is the only possible one and accepting it avoids paying for a second call. After reaping, the old token is rejected everywhere. Test: `test_completion_after_expiry_but_before_reap_is_accepted` together with `test_stale_worker_cannot_overwrite_new_attempt`.
2. **Cancelled runs stranding retry tasks. Fixed.** A retryable failure or a lease expiry on a cancelled run now moves the task to `cancelled`, and the run finalises. Tests: `test_cancel_then_retryable_failure_does_not_hang_run` and `test_cancel_then_lease_expiry_does_not_hang_run`.
3. **Provider cap vs live remote calls. Documented as a boundary.** The cap counts leases. A request whose worker lost its lease may still be running at the provider while the task is reassigned. See the "Global provider cap" bullet in `docs/service_design.md`. Enforcing this fully would need cancellable HTTP calls or a provider-side limit.
4. **Candidate window hiding other providers. Fixed.** Candidates are now ranked per provider (up to 10 each) instead of taking 25 globally. Test: `test_saturated_provider_does_not_hide_other_providers`.
5. **Opposite lock order in cancel. Fixed. It was a real deadlock.** The claim-vs-cancel interleaving raised `DeadlockDetected`. Cancel now commits its flag in a separate transaction, then cancels tasks and finalises in task → run order, the same order every other path uses. Test: `test_cancel_does_not_deadlock_with_concurrent_claim`.
6. **Fairness at capacity one. Improved, still documented as a heuristic.** With a cap of 1, ties always went to the oldest run. Migration `002_run_last_claimed.sql` adds `last_claimed_at`, and the fair scheduler breaks ties by least-recently-served. Test: `test_fair_scheduler_round_robins_at_capacity_one`. Strict priority can still starve lower priority, and candidate snapshots can go stale under concurrent claims; `service_design.md` says so.

Validation in the sandbox (Linux, Python 3.11, PostgreSQL 16): 51 tests passed, run three times. The experiments were rerun after these changes and `docs/service_experiments.json` was regenerated. Still pending: reproduction on the author's machine, a green CI run, and one real Ollama end-to-end run.
