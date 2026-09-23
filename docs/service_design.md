# ShiftWatch evaluation service: design and measured behaviour

The batch CLI runs an evaluation inside one process: if it dies at generation 180 of 240, the work is lost, and nothing prevents two copies of it from racing. The service turns the same evaluation code into a long-running system. A user submits a dataset, a model config and a rubric. Workers execute in the background. A crashed worker's tasks are picked up again. The user polls progress, and after a rubric change they re-score the stored outputs without calling the model again.

```
CLI client / OpenAPI UI (/docs)
        │  HTTP
        ▼
     FastAPI  ────────────  PostgreSQL  ◀──── claim / heartbeat / complete ────  Worker processes
  (submit, poll,          versions, tasks,                                          │ N slots each
   cancel, score)         attempts, outputs, scores                                 ▼
                                                                       Provider adapter (Ollama API)
                                                                       ├─ Ollama (real model)
                                                                       └─ mock provider (fault injection)
```

The design is a single API process plus independent workers, with no microservices and no message broker. Task state and results already live in PostgreSQL, so PostgreSQL also serves as the queue. That removes the classic "row committed but message not published" gap. A dedicated broker (with a transactional outbox) is worth adding only if measurements show the database is the bottleneck. At the loads measured below it uses about 2.8 to 3.6 commits per task, so it is not.

## 1. Data model: inference and scoring are separate

| Table | Mutability | Purpose |
| --- | --- | --- |
| `dataset_versions`, `dataset_cases` | immutable, content-addressed (SHA-256 of canonical JSON) | Uploading identical content returns the existing version, so edits always create a new version. |
| `model_configs` | immutable, content-addressed | provider + params (model, temperature, seed, base_url). `provider_key` is the concurrency bucket. |
| `inference_runs` | status only | One experiment: dataset version × model config, priority, `max_attempts`, optional `Idempotency-Key`. |
| `inference_tasks` | state machine | One (case, condition) prompt. Holds the lease. |
| `task_attempts` | append-only | One row per claim, and its id *is* the lease token. Records worker, outcome and error class. |
| `model_outputs` | write-once (PK `task_id`) | Raw text, parsed answer, parse error, latency, token usage. |
| `scoring_runs`, `scores` | append-only | Versioned rubric (hash of rubric + params + per-case overrides) and per-task scores. |

**Replay scoring vs. re-inference.** `POST /runs/{id}/scorings` reads `model_outputs` only. It never calls a provider (the test suite asserts that `task_attempts` does not grow). The same rubric on the same run is idempotent, and a changed rubric creates a new scoring run beside the old one. Per-case `case_overrides` (e.g. new `required_terms`) let a rubric fix be tested without a new dataset version. In the other direction, submitting a run always calls the model again: prompts are **not** de-duplicated across runs, because repeated sampling can be the purpose of the experiment.

**Malformed model replies are results, not failures.** If the model answers but breaks the JSON contract, the output is stored with `parse_error` and scored as incorrect. Retrying a temperature-0 model would just pay for the same answer again.

## 2. Task state machine

```
queued ──claim──▶ running ──complete (lease still valid)──▶ succeeded
                    │
                    ├─ retryable error, attempts left ─▶ retry_wait ──(backoff elapses)──▶ claimable
                    ├─ lease expired (worker died/hung) ▶ retry_wait   (failed if attempts exhausted)
                    ├─ fatal error / attempts exhausted ▶ failed
                    └─ error or lease expiry after the run was cancelled ▶ cancelled
queued | retry_wait ──run cancelled──▶ cancelled
```

A run becomes `succeeded` (all tasks succeeded), `failed` (none succeeded), `partially_failed` or `cancelled` once no task is queued, retrying or running. Partial completion is reported rather than hidden. Failed tasks are excluded from metrics and counted as `missing_outputs` on each scoring run.

**Error classes** (`providers.classify_exception`):

- Retryable: 429, 5xx, timeouts, connection resets, and a malformed HTTP envelope.
- Fatal: other 4xx, such as a wrong model name. Retrying cannot fix it.

Retries use exponential backoff with full jitter, capped at `SHIFTWATCH_RETRY_MAX_SECONDS`, and are bounded by `max_attempts`.

## 3. Failure handling: leases, attempts, write-once outputs

Claiming a task happens in one transaction (`SELECT … FOR UPDATE SKIP LOCKED`):

1. The task gets a fresh `lease_token` and `lease_expires_at = now() + lease`.
2. An attempt row is inserted.

All lease times use the database clock, so worker clock skew does not matter.

- **Who notices a crashed worker?** Every worker runs a heartbeat thread that extends its own leases every `lease/3`, and a reaper that returns expired `running` tasks to `retry_wait`. The expired attempt counts as used, because the provider may already have been called.
- **Two workers finish the same task. Whose result counts?** Completion is a conditional update: `… WHERE id = ? AND lease_token = ? AND status = 'running'`. The output insert happens in the same transaction. A worker whose lease was reaped gets `stale` back, and its attempt is recorded as `stale_discarded`. `model_outputs.task_id` is a primary key as a second line of defence.
- **Cancellation.** Queued and retrying tasks are cancelled immediately. A request already sent to a provider cannot be recalled and is already paid for, so it may finish and its output is kept. The run becomes `cancelled` when nothing is left running.
- **What this does *not* guarantee.** External calls are **at-least-once**. If a worker dies after the provider answered but before the commit, the retry calls the provider again. The database holds exactly one *effective* result per task, but not exactly one *call*. The crash experiment measures this directly.
- **Why an expired but not-yet-reaped lease is still honoured.** Completion, failure and heartbeat check the lease *token*, not the wall-clock expiry. The token is the fencing mechanism: until the reaper moves the task on, no other attempt can hold it, so accepting the original holder's result is safe and avoids paying for a second call. Once the reaper runs, the old token is invalid everywhere. A test covers both sides of this boundary.
- **Lock order.** Every path that touches both a task row and its run row locks the task first (claim, complete, fail, reap). Cancellation commits the cancel flag in its own transaction and only then cancels tasks and finalises the run, so it follows the same order. A regression test reproduces the claim-vs-cancel interleaving that used to deadlock.
- **Graceful shutdown.** On SIGTERM a worker stops claiming, keeps heartbeating, and lets in-flight tasks commit before it exits (compose sets `stop_grace_period`).

## 4. Concurrency control and backpressure

- **Global provider cap.** `provider_limits.max_concurrency` applies across all workers. It counts *leases*, not live sockets: if a worker loses its lease while its HTTP request is still in flight, the task can be reassigned and briefly overlap the old request at the provider. The cap is exact for healthy workers and approximate during lease loss; it is not an absolute remote-concurrency guarantee. Claimers for the same `provider_key` serialise on a transaction-scoped advisory lock, count running tasks, and claim only below the cap. Each provider is tried in its own short transaction, so no claimer holds two provider locks and lock ordering cannot deadlock. A crashed worker's tasks keep counting until their lease expires. That errs toward using less capacity rather than exceeding the cap.
- **Per-worker slots** (`--slots`) limit local threads. They are *not* a system-wide cap: 4 workers × 4 slots is 16 concurrent calls unless the provider cap says otherwise.
- **Backpressure.** Submission counts non-terminal tasks under an advisory lock. If the run would exceed `SHIFTWATCH_MAX_PENDING_TASKS`, the API returns **429 with `Retry-After`** instead of letting the backlog grow without bound. Single runs are also capped (`SHIFTWATCH_MAX_TASKS_PER_RUN`).
- **Fairness.** Priority comes first. Among equal-priority runs, the `fair` scheduler serves the run with the fewest running tasks, breaking ties by the run served least recently (`last_claimed_at`), so runs take turns even at a provider cap of 1. This is a heuristic, not a bounded-wait guarantee: strictly higher priority can still starve lower priority, and each claimer ranks candidates from a snapshot that concurrent claims may already have changed. `fifo` (oldest run first) is kept as the experimental baseline. Within a run, the lowest task id is claimed first, so requeued work resumes before untouched work.

## 5. Measured behaviour

Everything below was measured against the **mock provider**, not a real LLM. The mock is an Ollama-compatible HTTP server with seeded latency, 500s, dropped connections, invalid JSON and a 429 concurrency limit. These numbers describe the scheduler; they say nothing about model inference speed. Setup: a single machine, PostgreSQL 16, worker *processes* (so SIGKILL is a real crash), and the 240-task `llm_benchmark_60` dataset. Reproduce with `python scripts/service_experiments.py`; the raw numbers are in [`service_experiments.json`](service_experiments.json).

### Crash recovery

Setup: 2 workers × 4 slots, provider cap 8, lease 3 s, mock latency 300 ± 60 ms. After 96 outputs, one worker is SIGKILLed and a replacement starts immediately. 3 trials.

| Metric | Trial 0 | Trial 1 | Trial 2 |
| --- | --- | --- | --- |
| Tasks orphaned in `running` at kill | 4 | 4 | 4 |
| Lost tasks | 0 | 0 | 0 |
| Duplicate effective results | 0 | 0 | 0 |
| Time until every orphaned task had a committed output | 4.63 s | 4.67 s | 4.59 s |
| Provider calls / extra (duplicate) calls | 244 / 4 | 244 / 4 | 244 / 4 |
| Run status | succeeded | succeeded | succeeded |

The 4 extra provider calls are exactly the 4 orphaned in-flight requests. This is the at-least-once boundary made visible. Recovery time is roughly lease (3 s) + reaper interval (≤ 2 s) + one call. A shorter lease recovers faster but risks falsely expiring a slow but healthy worker. That is a tunable trade-off, not a free win.

*A measured fix:* the first version claimed tasks in `available_at` order. Reaped tasks were stamped `now()`, so they went to the back of the queue, and recovery took about 7.6 s (roughly the rest of the run). Switching to lowest-id-first cut recovery to about 4.6 s with no change to total makespan.

### Scaling with worker count

Setup: 3% injected HTTP 500s, mock latency 200 ± 40 ms, provider cap not binding.

| Workers × slots | Makespan | Throughput | p95 queue wait | DB commits / task | Attempt error rate | Max in-flight at provider |
| --- | --- | --- | --- | --- | --- | --- |
| 1 × 4 | 12.61 s | 19.0 tasks/s | 11.7 s | 3.6 | 0.8% | 4 |
| 2 × 4 | 6.49 s | 37.0 tasks/s | 5.9 s | 3.1 | 0.8% | 8 |
| 4 × 4 | 3.84 s | 62.5 tasks/s | 3.3 s | 2.8 | 0.8% | 16 |

The ideal makespan (240 × 0.2 s / slots) is 12 s, 6 s and 3 s. The gap at 16 slots (3.84 s vs 3 s) is claim-transaction serialisation on the single provider lock, plus polling when the queue drains. Every run completed, with no lost tasks and no duplicate results.

### Overload and fairness

Setup: a large run (240 tasks) is submitted, then a small run (16 tasks) 1 s later. 1 worker × 8 slots. The mock returns 429 above 4 concurrent requests.

| Scheduler | Provider cap | Small-run makespan | Small-run p95 wait | Large-run makespan | Provider 429s | Tasks failed | Max in-flight at provider |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fifo | 4 | 12.34 s | 12.1 s | 12.57 s | 0 | 0 | 4 |
| fair | 4 | **1.87 s** | 1.5 s | 13.39 s | 0 | 0 | 4 |
| fair | 16 (above the real limit) | 8.48 s | 0.1 s | 13.10 s | 1418 | 24 | 6 |

Fair-share cuts the small run's completion time about 6.6× and costs the large run about 0.8 s. When the configured cap matches the provider's real limit, there are zero 429s. When it overshoots, the provider rejects 1,418 requests, the retry budget burns, and 24 tasks fail even though the provider was never down. The global cap is what prevents that.

## Known limitations and next steps

- There is no authentication, and `params.base_url` lets any API caller point workers at an arbitrary URL. Keep the API on a private network, or add auth and an allow-list of provider endpoints before exposing it.
- The provider cap is a concurrency limit, not a requests-per-minute or token budget. A token-bucket limit would need a shared counter table or Redis.
- Crashed workers hold cap slots until their lease expires.
- Claims poll PostgreSQL (`poll_seconds`). `LISTEN/NOTIFY` would cut idle latency. A broker becomes worth it only if claim contention dominates at much higher worker counts.
- The API opens one DB connection per request. Add `psycopg_pool` if connection setup shows up in latency profiles.
- The fault-injection numbers come from one machine and a mock. A real Ollama end-to-end run should be reported separately, with its own hardware notes.
- Only the LLM keyword rubric is exposed through the service. The code-agent and SQL evaluators would slot in as additional task/rubric types.
