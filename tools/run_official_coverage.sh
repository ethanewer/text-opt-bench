#!/usr/bin/env bash
# Official, reproducible coverage runner for text-opt-bm.
#
# CANONICAL SETUP (identical for every config so runs are directly comparable):
#   model gpt-5.5, 1-hour wall box (3600 s), 40-iteration cap,
#   codex-timeout 1200 s, concurrency 16, 5 independent runs per task.
#
# The three official coverage configs (run each; --only-missing skips any run
# dir already complete, so reruns are cheap and idempotent):
#   1. none   : all 14 core tasks, effort=none  (weak / non-reasoning model)
#   2. low    : all 14 core tasks, effort=low   (strong model)
#   3. lowvv  : the 5 generalization *_exposed variants, effort=low
#               (val data made visible — the overfitting arm; run
#                tools/make_exposed_variants.py first to create the variants).
#
# Existing runs at this exact setup already satisfy some configs and are reused
# (see docs/coverage_results.md for the config -> campaign-prefix mapping):
#   low  <- 5xE- (11 original tasks) + GENF2- (3 new generalization tasks)
#   none <- CMPN- (8 tasks); the other 6 are filled here under cov-none-
#
# Usage:
#   tools/run_official_coverage.sh --effort none  --prefix cov-none-  --tasks <list>
#   tools/run_official_coverage.sh --effort low   --prefix cov-lowvv- --tasks <exposed list>
cd "$(dirname "$0")/.."
exec python3.12 tools/run_campaign.py \
  --runs 5 --concurrency 16 --timebox 3600 --iterations 40 \
  --model gpt-5.5 --codex-timeout 1200 --only-missing "$@"
