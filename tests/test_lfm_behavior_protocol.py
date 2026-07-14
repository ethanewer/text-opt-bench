"""Model-free checks for the LFM behavioral-compression protocol."""

import json
from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench import heldout, runner
from bench.ifbench_subset import configure_nltk_data, loose_pass
from bench.lfm_behavior_compression import (
    bfcl_pass,
    parse_tool_calls,
    response_cap,
    verify_ifbench_assets,
)
from bench.slm_private import require_private_slm_operator_state_absent


class TokenizerProbe:
    def __call__(self, value, add_special_tokens=False):
        del add_special_tokens
        return type("Tokens", (), {"input_ids": list(range(int(value)))})()


def main():
    task = "slm_weight_compression_lfm25"
    config = runner.load_config(task)
    assert config["protocol_version"] == 4
    assert config["metric"].startswith("mean BF16 behavioral regression")
    assert config["deferred_aggregation"] == "lfm_behavior_single_shard"
    assert "slm_weight_compression_lfm25_regression" not in runner.list_tasks()

    data = runner.task_dir(task) / "data"
    verify_ifbench_assets(data)
    import nltk

    previous = configure_nltk_data(data / "ifbench_nltk_data")
    try:
        assert nltk.data.path == [str(data / "ifbench_nltk_data")]
        for resource in (
            "corpora/stopwords",
            "taggers/averaged_perceptron_tagger_eng",
            "tokenizers/punkt_tab/english",
        ):
            nltk.data.find(resource)
        validation = heldout.read(data / "heldout_val.bin")
        test = heldout.read(data / "heldout_test.bin")["regression"]
        rows = [
            row for payload in (validation, test)
            for row in payload["datasets"]["ifbench"]
        ]
        assert len(rows) == 40
        assert all(loose_pass(row, row["bf16_response"]) for row in rows)
    finally:
        nltk.data.path[:] = list(previous)

    probe = TokenizerProbe()
    assert response_cap(probe, {"bf16_response": "1"}, 128) == 20
    assert response_cap(probe, {"bf16_response": "16"}, 128) == 20
    assert response_cap(probe, {"bf16_response": "17"}, 128) == 40
    assert response_cap(probe, {"bf16_response": "111"}, 128) == 128
    expected = [{"name": "lookup", "arguments": {"key": "value", "n": 2}}]
    assert parse_tool_calls("lookup(key='value', n=2)") == expected
    assert bfcl_pass("lookup(key='value', n=2)", expected)
    assert not bfcl_pass("lookup('value')", expected)
    assert not bfcl_pass("obj.lookup(key='value', n=2)", expected)

    with tempfile.TemporaryDirectory() as tmp:
        marker = Path(tmp) / "private.json"
        marker.write_text(json.dumps({"secret": True}))
        try:
            require_private_slm_operator_state_absent(marker)
        except RuntimeError as exc:
            assert str(marker) in str(exc)
        else:
            raise AssertionError("private-state guard accepted an exposed artifact")
    require_private_slm_operator_state_absent()
    print("LFM behavioral protocol checks passed")


if __name__ == "__main__":
    main()
