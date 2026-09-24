-- Round-robin tie-breaker for the fair scheduler: among equal-priority runs with the
-- same number of running tasks, serve the one that was served least recently.
ALTER TABLE inference_runs ADD COLUMN last_claimed_at timestamptz;
