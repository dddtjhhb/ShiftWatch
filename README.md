# ShiftWatch

ShiftWatch is a reproducible evaluation pipeline for studying how language-model reliability changes under controlled input distribution shifts. Its main direction is LLM behavioral evaluation; a transparent text-classification baseline remains as a pipeline validation case study.

The project now also includes an initial code-agent regression-testing module. It measures whether generated Python implementations continue to pass executable tests when task prompts receive irrelevant context, false premises, or long benign context.

This project is motivated by trustworthy ML and AI safety evaluation. It does **not** claim to solve AI alignment. Its narrower goal is to make model failures measurable, reproducible, and visible before deploying more complex training methods.

## Research questions

- How much does model accuracy degrade under typos, truncation, irrelevant context, and prompt injection?
- Does a model abstain when evidence is insufficient, or remain confidently wrong?
- Are reported confidence scores calibrated under shift?
- How quickly can CUSUM or EWMA detect an increase in the model's failure rate?
- Later: does parameter-efficient fine-tuning improve average accuracy at the cost of out-of-distribution reliability?

## Current scope: Phase 1

The first phase is intentionally CPU-only and dependency-light:

- validated JSONL dataset contract;
- deterministic perturbation generation;
- transparent keyword baseline;
- accuracy, error, abstention, confidence, and Brier-score metrics;
- clean-versus-shift comparison;
- CUSUM and EWMA degradation detectors;
- CSV output for failure analysis;
- automated tests and deterministic seeds.

The keyword model is not presented as a competitive classifier. It is a transparent baseline that verifies the evaluation system before an LLM is added.

## LLM behavioral evaluation

The LLM path asks the same factual question under four conditions: clean wording, a paraphrase, irrelevant context, and a user-supplied false premise. Each backend returns a structured answer, confidence score, and abstention decision. ShiftWatch measures correctness, behavioral consistency, false-premise refutation, and confidently wrong answers.

The included `fixture` backend contains recorded responses only so tests and GitHub Actions stay deterministic. It is explicitly not a real model result. A real local model can be served through Ollama:

```bash
python -m shiftwatch.cli llm-evaluate datasets/llm_demo.jsonl \
  --backend ollama --model llama3.2:3b \
  --output results/llm_evaluation.csv
```

For a larger experiment, the versioned starter benchmark contains 60 cases and four prompt conditions per case (240 generations). Free-response mode keeps the model's full explanatory paragraph inside the structured result so it can be reviewed as well as scored:

```bash
python scripts/build_llm_benchmark.py
python -m shiftwatch.cli llm-evaluate datasets/llm_benchmark_60.jsonl \
  --backend ollama --model llama3:latest --response-mode free \
  --output results/llama3_benchmark_60_free.csv
```

The benchmark spans factual knowledge, reasoning, computing, statistics, monitoring, ML, AI safety concepts, and ten intentionally unknowable questions. Metrics are broken down by prompt condition and category. The CSV preserves every full answer and records answer length, mentions of forbidden terms, explicit versus semantic abstention, false-premise refutation, and high-confidence errors. Mentioning an incorrect term is an audit signal rather than an automatic failure because a good correction may quote the false claim. The starter cases are suitable for pipeline development, not a publishable benchmark; facts and rubrics require independent review before making research claims.

Previously generated responses can be rescored after improving a rubric without paying the inference cost again:

```bash
python -m shiftwatch.cli llm-evaluate datasets/llm_benchmark_60.jsonl \
  --backend recorded --responses results/llama3_benchmark_60_free.csv \
  --output results/llama3_benchmark_60_rescored.csv
```

The same adapter boundary can later point to an Ollama server running on a university GPU node. Cluster access, job scheduling, and model storage depend on the account and allocation supplied by the university, course, or research group.

## Evaluation service (recoverable, traceable, concurrent)

**Integration status:** the supplied service patch is integrated locally. Core tests pass, but PostgreSQL integration and fault experiments have not been rerun in this environment. Recovery/concurrency claims below describe the intended design; see [integration issues](docs/service_integration_issues.md) for known gaps before relying on them.

The batch commands above run in a single process. `shiftwatch.service` runs the same LLM evaluation as a small backend system: a FastAPI server plus independent worker processes, with PostgreSQL holding immutable dataset/model-config versions, the task queue, every execution attempt, write-once model outputs, and versioned scoring runs. Workers hold time-limited leases with heartbeats. If a worker crashes, its tasks are requeued. A stale worker cannot overwrite a newer result. A global per-provider concurrency cap, queue-capacity backpressure (HTTP 429) and fair-share scheduling keep one large experiment from overloading the model server or starving small ones. Re-scoring with a revised rubric reads stored outputs and never calls the model.

```bash
docker compose up --build          # API :8000, 2 workers, Postgres, fault-injecting mock provider
python -m shiftwatch.service client submit datasets/llm_benchmark_60.jsonl \
  --model mock --base-url http://mock-provider:11500 --wait
python -m shiftwatch.service client score RUN_ID --overrides revised_terms.json
# Real model: docker compose --profile ollama up, then --model llama3:latest --base-url http://ollama:11434
```

Without Docker: `pip install -e '.[service]'`, set `SHIFTWATCH_DATABASE_URL`, then run `python -m shiftwatch.service migrate`, `... api`, `... worker --slots 4`. Interactive API docs are served at `/docs`.

The supplied patch includes a crash-recovery and performance report. Those measurements have not been independently reproduced here and must not be presented as newly verified results. External calls may be repeated after failures; the database is intended to accept at most one effective output per task. See [`docs/service_design.md`](docs/service_design.md) for the design and supplied experiments.

Integration tests need a throwaway PostgreSQL database: `SHIFTWATCH_TEST_DATABASE_URL=postgresql://... python -m unittest tests.test_service`. They are skipped when that variable is unset.

## Code-agent evaluation

The `code-evaluate` command accepts a compact HumanEval/MBPP-style JSONL contract: task id, prompt, entry point, executable assertions, a task-specific false premise, and optional deterministic fixture candidates. Each task is evaluated under four conditions and can be repeated to measure run-to-run stability:

- `clean`: original task;
- `irrelevant_context`: unrelated project information is prepended;
- `false_premise`: a task-specific incorrect implementation assumption is supplied;
- `long_context`: benign instructions are repeated before the task.

```bash
python -m shiftwatch.cli code-evaluate datasets/code_tasks_demo.jsonl \
  --backend fixture --repeats 3 \
  --output results/code_evaluation.csv

# Real local model (slow; requires Ollama and an installed model)
python -m shiftwatch.cli code-evaluate datasets/code_tasks_demo.jsonl \
  --backend ollama --model llama3:latest --repeats 3 \
  --output results/llama3_code_evaluation.csv
```

Reports include pass rate, task-level bootstrap confidence intervals, abstention, generation latency, optional cost, repeated-run outcome stability, and an executable failure taxonomy (`syntax_error`, `missing_entry_point`, `policy_rejection`, `timeout`, `assertion_failure`, and `runtime_error`). Full candidate code is retained for manual failure review.

The bundled four-task dataset is a software test and demonstration, not a research benchmark. The next study milestone is a frozen, independently reviewed 30-50 task subset from a public benchmark, two model or agent configurations, 3-5 runs per task, and manual taxonomy assignment for 20-30 failures.

An initial 16-generation local Llama 3 smoke test passed 4/4 clean, 4/4 irrelevant-context, 2/4 false-premise, and 4/4 long-context cases. Mean generation latency increased from 2.62 seconds on clean prompts to 11.65 seconds under the synthetic long context. These numbers validate the pipeline only; see [`docs/code_agent_smoke_analysis.md`](docs/code_agent_smoke_analysis.md) for the two manually reviewed failures and limitations.

### Execution safety

Generated code is untrusted. ShiftWatch rejects imports, dynamic evaluation, file access calls, dunder attributes, and missing entry points before execution. Accepted candidates run with isolated Python, a temporary working directory, a wall-clock timeout, and best-effort Unix resource limits. These controls reduce risk but are **not a security boundary**. Do not run untrusted model output on a personal or production machine; use a disposable container or VM for real benchmark runs.

## SQL misconception diagnosis

The SQL extension evaluates whether a model can identify the underlying concept gap in a semantically incorrect query without giving away corrected SQL. The versioned starter set contains 30 labeled cases spanning NULL semantics, grouping, aggregation grain, join multiplicity, window frames, subquery cardinality, boolean precedence, temporal boundaries, alias scope, set semantics, and data-modification safety.

Each query is evaluated under clean, paraphrased, irrelevant-context, and false-premise instructions. The model returns structured concept labels, a high-level hint, confidence, and abstention while `suggested_sql` must remain null.

```bash
python scripts/build_sql_benchmark.py
python -m shiftwatch.cli sql-evaluate datasets/sql_misconceptions_30.jsonl \
  --backend fixture --snapshot-id fixture-v1 \
  --output results/sql_evaluation.csv \
  --concept-output results/sql_concept_metrics.csv

# Small real-model smoke test
python -m shiftwatch.cli sql-evaluate datasets/sql_misconceptions_30.jsonl \
  --backend ollama --model llama3:latest --max-tasks 5 \
  --snapshot-id llama3-smoke
```

The report includes concept-level micro precision/recall, answer-leakage rate, abstention, latency, cost, and exact error-attribution consistency across prompt conditions. Leakage is flagged when a model fills `suggested_sql`, emits a code block, or includes a task-specific repair fragment.

`sql_concept_metrics.csv` exports recall error by concept and snapshot. After appending ordered model or prompt snapshots, one concept can be monitored with the existing sequential detectors:

```bash
python -m shiftwatch.cli sql-monitor results/sql_concept_metrics_history.csv \
  --concept null_semantics --target 0.10 \
  --output results/sql_null_semantics_alarms.csv
```

This is currently a research extension and evaluation scaffold, not a completed comparative study. Before making research claims, the labels and leakage rubrics require independent review, multiple model or prompt configurations, repeated runs, and manual adjudication of disagreements.

A 20-diagnosis Llama 3 smoke test on the first five cases validated the closed-vocabulary path and produced zero detected answer leakage. See [`docs/sql_diagnostic_smoke_analysis.md`](docs/sql_diagnostic_smoke_analysis.md) for the preliminary metrics, the initial label-mismatch failure, and limitations. The bundled `demo_sql_concept_history.csv` is synthetic data used only to verify that concept-level degradation reaches CUSUM/EWMA.

Automated exact-label and leakage checks are only screening tools. The human-review path creates a randomly ordered packet that hides task ids, perturbation conditions, benchmark labels, and model identity. A reviewer records semantic concept labels, overall correctness, answer leakage, explanation quality, ambiguity, and confidence before a separate key is used to unblind and summarize the study. See [`docs/sql_human_review_protocol.md`](docs/sql_human_review_protocol.md) for the rubric and commands.

## Architecture

```text
versioned dataset
      |
controlled perturbations
      |
model adapter -> structured Prediction
      |
reliability metrics + failure table
      |
code task -> controlled prompt shift -> guarded test execution
      |
pass rate + bootstrap CI + failure taxonomy + stability
      |
batch error-rate time series
      |
CUSUM / EWMA degradation alarms
```

## Quick start

Python 3.10+ is sufficient.

```bash
python -m shiftwatch.cli evaluate datasets/demo.jsonl --output results/evaluation.csv
python -m shiftwatch.cli llm-evaluate datasets/llm_demo.jsonl \
  --backend fixture --output results/llm_evaluation.csv
python -m shiftwatch.cli monitor datasets/demo_batch_metrics.csv \
  --target 0.05 --output results/alarms.csv
python -m unittest discover -v
```

The CLI prints a JSON summary and writes one row per example and condition. Fixed random seeds make perturbations reproducible.

The monitoring command expects an ordered CSV series:

```csv
batch_id,error_rate
2026-08-01,0.05
2026-08-02,0.06
```

Each batch might represent a day, deployment window, or fixed number of predictions. ShiftWatch does not treat perturbation conditions as time points: batch error rates must come from genuinely ordered evaluation or production windows. The target is supplied explicitly so the monitoring baseline is documented rather than estimated from the same period being tested.

## Run without a local terminal

The `ShiftWatch` GitHub Actions workflow provides a browser-based entry point:

1. Open the repository's **Actions** tab.
2. Select **ShiftWatch**.
3. Choose **Run workflow**.
4. Select `evaluate` or `monitor`; monitoring also accepts a target error rate.
5. After the run finishes, download the `shiftwatch-*` artifact from the run page.

Every push and pull request also runs the automated test suite. Manual runs execute tests first, then generate a CSV result and a JSON summary. Artifacts are retained for 14 days and require no credentials or external model API.

## Dataset format

One manually reviewed example per line:

```json
{"id":"example-001","text":"Scientists published a laboratory study.","label":"science"}
```

Allowed labels are `technology`, `business`, `science`, `campus`, and `other`. The demo data exists only to exercise the pipeline and must not be reported as a meaningful benchmark.

## Metrics

- **Accuracy/error rate:** task performance, counting abstention as unresolved.
- **Abstain rate:** how often the model declines to classify.
- **Brier score:** squared error between confidence and correctness; lower is better.
- **Error increase vs. clean:** degradation attributable to each controlled condition.
- **Detection alarms:** points at which CUSUM or EWMA flags sustained error growth.

## Responsible interpretation

The current dataset is tiny and synthetic, so results are software checks rather than scientific conclusions. A valid study will require a documented sampling procedure, independently reviewed labels, class-balance reporting, frozen test data, multiple model runs, uncertainty intervals, and qualitative review of failure cases.

## Roadmap

1. Build and document a 200–500 example human-reviewed evaluation set.
2. Add a local Ollama adapter with strict JSON parsing and retry accounting.
3. Add paraphrase and out-of-domain shifts without changing gold labels.
4. Report bootstrap confidence intervals and per-class false positives/negatives.
5. Simulate gradual and abrupt degradation; compare detection delay and false alarms.
6. Add GPU batch inference only when CPU inference becomes the bottleneck.
7. Add LoRA/QLoRA as an experiment and compare pre/post-tuning calibration, abstention, and shifted performance.

## Project positioning

ShiftWatch is best described as an AI reliability evaluation and monitoring project spanning reproducible data systems, trustworthy ML, and sequential change detection. Fine-tuning is a later experimental intervention, not evidence of alignment by itself.
