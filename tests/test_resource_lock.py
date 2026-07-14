"""Checks that campaign locks gate evaluations, not optimization loops."""

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench import runner
from bench.resource_lock import (LIMITS_ENV, LOCK_DIR_ENV, REQUESTS_ENV,
                                 WAIT_LOG_ENV, evaluation_slot,
                                 evaluation_slots,
                                 record_wait_interval)
from tools.run_campaign import eval_queue_seconds


def main():
    child = (
        "from bench.resource_lock import evaluation_slot; import sys, time\n"
        "with evaluation_slot(sys.argv[1], priority=sys.argv[2]) as wait:\n"
        " print(str(wait) + ' ' + str(time.time()), flush=True); time.sleep(.05)\n"
    )
    with tempfile.TemporaryDirectory(prefix="textopt_eval_locks_") as tmp:
        limits = json.dumps({"cpu": 1, "accelerator": 1})
        wait_log = str(Path(tmp) / "eval_queue.jsonl")
        env = dict(os.environ, **{LOCK_DIR_ENV: tmp, LIMITS_ENV: limits,
                                  WAIT_LOG_ENV: wait_log})
        old_dir, old_limits = os.environ.get(LOCK_DIR_ENV), os.environ.get(LIMITS_ENV)
        old_wait_log = os.environ.get(WAIT_LOG_ENV)
        os.environ.update({LOCK_DIR_ENV: tmp, LIMITS_ENV: limits,
                           WAIT_LOG_ENV: wait_log})
        try:
            launch_start = time.time()
            with evaluation_slot("cpu"):
                blocked = subprocess.Popen(
                    [sys.executable, "-c", child, "cpu", "foreground"], cwd=ROOT, env=env,
                    stdout=subprocess.PIPE, text=True,
                )
                overlap = subprocess.run(
                    [sys.executable, "-c", child, "accelerator", "foreground"], cwd=ROOT,
                    env=env, capture_output=True, text=True, timeout=2,
                )
                time.sleep(.2)
                assert blocked.poll() is None, "same-resource evaluation did not queue"
                assert float(overlap.stdout.split()[0]) < .15, "independent resource did not overlap"
                live_refund = eval_queue_seconds(wait_log, launch_start)
                assert live_refund >= .15, "active queue interval was not refunded live"
            stdout, _ = blocked.communicate(timeout=2)
            assert float(stdout.split()[0]) >= .15, "queued evaluation reported no wait"
            final_refund = eval_queue_seconds(wait_log, launch_start)
            assert final_refund >= .15, "completed queue interval was not refunded"

            # A trusted parent can add an inner shared-MPS lease wait after the
            # evaluator reports its authenticated timestamps.
            inner_started = time.time()
            time.sleep(.03)
            inner_ended = time.time()
            before_inner = eval_queue_seconds(wait_log, launch_start)
            record_wait_interval(
                inner_started, inner_ended, category="slm-mps-lock")
            after_inner = eval_queue_seconds(wait_log, launch_start)
            assert after_inner >= before_inner + .02, (
                "inner MPS lock wait was not refunded")

            # A foreground evaluation jumps ahead of already-waiting
            # background holdout work as soon as the occupied slot opens.
            with evaluation_slot("accelerator"):
                background = subprocess.Popen(
                    [sys.executable, "-c", child, "accelerator", "background"],
                    cwd=ROOT, env=env, stdout=subprocess.PIPE, text=True,
                )
                time.sleep(.1)
                foreground = subprocess.Popen(
                    [sys.executable, "-c", child, "accelerator", "foreground"],
                    cwd=ROOT, env=env, stdout=subprocess.PIPE, text=True,
                )
                time.sleep(.1)
                assert background.poll() is None and foreground.poll() is None
            foreground_out, _ = foreground.communicate(timeout=2)
            background_out, _ = background.communicate(timeout=2)
            foreground_acquired = float(foreground_out.split()[1])
            background_acquired = float(background_out.split()[1])
            assert foreground_acquired < background_acquired, (
                "background holdout work bypassed a foreground waiter")

            # Weighted FIFO: while two of four units are occupied, a four-unit
            # request at the head must wait. A later one-unit request cannot
            # backfill past it and starve the expensive task. Once capacity is
            # free, the heavy task runs alone and the cheap task follows.
            weighted_child = (
                "from bench.resource_lock import evaluation_slot; import sys,time\n"
                "with evaluation_slot('cpu') as wait:\n"
                " print(str(wait)+' '+str(time.time()),flush=True);"
                " time.sleep(float(sys.argv[1]))\n"
            )
            weighted_limits = json.dumps({"cpu": 4, "accelerator": 1})
            base = dict(os.environ, **{
                LOCK_DIR_ENV: tmp, LIMITS_ENV: weighted_limits,
                WAIT_LOG_ENV: wait_log,
            })
            blocker_env = dict(base, **{REQUESTS_ENV: json.dumps({"cpu": 2})})
            blocker = subprocess.Popen(
                [sys.executable, "-c", weighted_child, ".25"], cwd=ROOT,
                env=blocker_env, stdout=subprocess.PIPE, text=True)
            blocker.stdout.readline()  # two units are now occupied
            heavy_env = dict(base, **{REQUESTS_ENV: json.dumps({"cpu": 4})})
            heavy = subprocess.Popen(
                [sys.executable, "-c", weighted_child, ".1"], cwd=ROOT,
                env=heavy_env, stdout=subprocess.PIPE, text=True)
            time.sleep(.07)  # ensure the heavy request owns the earlier ticket
            cheap_env = dict(base, **{REQUESTS_ENV: json.dumps({"cpu": 1})})
            cheap = subprocess.Popen(
                [sys.executable, "-c", weighted_child, "0"], cwd=ROOT,
                env=cheap_env, stdout=subprocess.PIPE, text=True)
            heavy_out, _ = heavy.communicate(timeout=3)
            cheap_out, _ = cheap.communicate(timeout=3)
            blocker.communicate(timeout=3)
            heavy_acquired = float(heavy_out.split()[1])
            cheap_acquired = float(cheap_out.split()[1])
            assert heavy_acquired < cheap_acquired, (
                "cheap weighted request bypassed the FIFO head")
            assert float(heavy_out.split()[0]) >= .12, (
                "four-unit request acquired without four free units")

            # One evaluator can reserve its accelerator and a weighted CPU
            # share together. It must wait for CPU after acquiring the free
            # accelerator, and keep both leases for the evaluator body.
            multi_env = dict(base, **{
                REQUESTS_ENV: json.dumps({"accelerator": 1, "cpu": 2})})
            multi_child = (
                "from bench.resource_lock import evaluation_slots; import time\n"
                "with evaluation_slots('accelerator') as wait:\n"
                " print(str(wait),flush=True); time.sleep(.05)\n"
            )
            holding_env = dict(base, **{
                REQUESTS_ENV: json.dumps({"cpu": 3})})
            holding = subprocess.Popen(
                [sys.executable, "-c", weighted_child, ".25"], cwd=ROOT,
                env=holding_env, stdout=subprocess.PIPE, text=True)
            holding.stdout.readline()
            multi = subprocess.Popen(
                [sys.executable, "-c", multi_child], cwd=ROOT,
                env=multi_env, stdout=subprocess.PIPE, text=True)
            time.sleep(.12)
            assert multi.poll() is None, (
                "multi-resource evaluator ignored its CPU request")
            holding.communicate(timeout=3)
            multi_out, _ = multi.communicate(timeout=3)
            assert float(multi_out.strip()) >= .1
        finally:
            if old_dir is None:
                os.environ.pop(LOCK_DIR_ENV, None)
            else:
                os.environ[LOCK_DIR_ENV] = old_dir
            if old_limits is None:
                os.environ.pop(LIMITS_ENV, None)
            else:
                os.environ[LIMITS_ENV] = old_limits
            if old_wait_log is None:
                os.environ.pop(WAIT_LOG_ENV, None)
            else:
                os.environ[WAIT_LOG_ENV] = old_wait_log

    # Runner integration: a nonce-authenticated MPS evaluator result has its
    # canonical inner-lock wait added to both result telemetry and the campaign
    # refund log by the trusted parent, without exposing that log to the child.
    with tempfile.TemporaryDirectory(prefix="textopt_inner_mps_wait_") as raw:
        raw = Path(raw)
        task = raw / "fixture_mps"
        task.mkdir()
        (task / "config.json").write_text(json.dumps({
            "evaluation_resource": "accelerator", "required_device": "mps",
            "cpu_s": 5, "timeout_s": 5,
        }))
        (task / "evaluate.py").write_text(
            "import time\n"
            "from bench import eval_lib\n"
            "from bench.slm_mps_lock import canonical_mps_lock_identity\n"
            "now=time.time()\n"
            "lock={**canonical_mps_lock_identity(),"
            "'wait_started_unix':now-.03,'acquired_unix':now,"
            "'wait_seconds':.03}\n"
            "eval_lib.succeed(0.0,{'exclusive_mps_lock':lock})\n")
        program = raw / "program.py"
        program.write_text("pass\n")
        log = raw / "waits.jsonl"
        old_tasks = runner.TASKS_DIR
        old_wait_log = os.environ.get(WAIT_LOG_ENV)
        runner.TASKS_DIR = raw
        os.environ[WAIT_LOG_ENV] = str(log)
        try:
            launched = time.time() - 1
            result = runner.evaluate("fixture_mps", program)
            assert result["ok"] and result["eval_queue_seconds"] >= .025
            assert eval_queue_seconds(log, launched) >= .025
        finally:
            runner.TASKS_DIR = old_tasks
            if old_wait_log is None:
                os.environ.pop(WAIT_LOG_ENV, None)
            else:
                os.environ[WAIT_LOG_ENV] = old_wait_log
    print("resource lock checks passed")


if __name__ == "__main__":
    main()
