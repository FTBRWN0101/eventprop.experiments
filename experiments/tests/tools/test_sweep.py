"""The sweep runner: grid shape, resume, and the safeguard override.

No subprocess is ever started here. run_arm is the only thing that shells out and it is
monkeypatched, so these run in the plain .venv in milliseconds.
"""

import json

import pytest

from tools import sweep


def test_the_grid_is_the_product_of_its_axes():
    grid = sweep.build_grid()
    assert len(grid) == (len(sweep.ENCODINGS) * len(sweep.ALGORITHMS)
                         * len(sweep.SEEDS) * len(sweep.WINDOWS))
    #every cell appears exactly once
    assert len({a.key for a in grid}) == len(grid)


def test_the_grid_is_deterministic():
    assert [a.key for a in sweep.build_grid()] == [a.key for a in sweep.build_grid()]


def test_seeds_are_contiguous_from_one():
    #a non-contiguous set would invite the question of which seeds were dropped
    assert sweep.SEEDS == tuple(range(1, len(sweep.SEEDS) + 1))
    assert len(sweep.SEEDS) >= 8, "the metrics contract asks for at least 8 seeds"


def test_latency_alone_gets_the_silent_neuron_override():
    """D67 aborts at 10% silent; only latency is knowingly run through the breach."""
    for arm in sweep.build_grid(seeds=(1,), windows=(5,)):
        command = sweep.arm_command(arm, "py", "rv_fwd", "vix9d", "weekly", "snn")
        override = "--allow-silent-neurons" in command
        assert override == (arm.encoding == "latency"), arm.key


def test_the_command_carries_every_axis_of_the_arm():
    arm = sweep.Arm(encoding="delta", algorithm="eprop", seed=4, input_window=45)
    command = sweep.arm_command(arm, "py", "rv_fwd", "vix9d", "weekly", "snn,har_rv")
    for flag, value in (("--encoding", "delta"), ("--algorithm", "eprop"),
                        ("--seed", "4"), ("--input-window", "45"),
                        ("--target", "rv_fwd"), ("--iv-leg", "vix9d"),
                        ("--models", "snn,har_rv")):
        assert command[command.index(flag) + 1] == value, flag


def test_result_dir_is_recovered_from_run_output():
    stdout = "noise\nresults written to C:\\runs\\20260818-1200_weekly\nmore noise\n"
    assert sweep.result_dir(stdout, "") == "C:\\runs\\20260818-1200_weekly"


def test_result_dir_is_found_on_stderr_too():
    """The harness reports through logging, whose default handler is stderr. Reading
    stdout alone left every manifest record without its directory.
    """
    stderr = "14:01:12 results written to C:\\runs\\20260818-1200_weekly\n"
    assert sweep.result_dir("", stderr) == "C:\\runs\\20260818-1200_weekly"


def test_result_dir_is_blank_when_the_run_reported_none():
    assert sweep.result_dir("nothing useful here", "nor here") == ""


def _records(*entries):
    return {key: {"key": key, "returncode": code} for key, code in entries}


def _fake_record(arm, returncode=0):
    """What run_arm really returns, so the doubles match the contract they stand in for."""
    from dataclasses import asdict

    record = {"key": arm.key, **asdict(arm), "returncode": returncode,
              "seconds": 0.0, "run_dir": ""}
    if returncode != 0:
        record["error"] = "boom"
    return record


def test_completed_arms_are_skipped_on_resume():
    grid = sweep.build_grid(encodings=("rate",), algorithms=("eventprop",),
                            seeds=(1, 2, 3), windows=(5,))
    done = _records(("rate_eventprop_L5_s1", 0))
    todo = sweep.pending(grid, done, retry_failed=False)
    assert [a.seed for a in todo] == [2, 3]


def test_failed_arms_are_skipped_unless_retrying():
    grid = sweep.build_grid(encodings=("rate",), algorithms=("eventprop",),
                            seeds=(1, 2), windows=(5,))
    done = _records(("rate_eventprop_L5_s1", 1))
    assert [a.seed for a in sweep.pending(grid, done, retry_failed=False)] == [2]
    assert [a.seed for a in sweep.pending(grid, done, retry_failed=True)] == [1, 2]


def test_manifest_round_trips(tmp_path):
    path = tmp_path / "m.json"
    records = {"a": {"key": "a", "returncode": 0, "seconds": 1.0}}
    sweep.save_manifest(path, records)
    assert sweep.load_manifest(path) == records


def test_an_unreadable_manifest_starts_fresh_rather_than_crashing(tmp_path):
    path = tmp_path / "m.json"
    path.write_text("{not json", encoding="utf8")
    assert sweep.load_manifest(path) == {}


def test_a_missing_manifest_is_empty(tmp_path):
    assert sweep.load_manifest(tmp_path / "absent.json") == {}


def test_the_manifest_is_written_after_every_arm(tmp_path, monkeypatch):
    """A kill mid-sweep must lose at most one run, so persistence cannot wait."""
    seen = []

    def fake_run_arm(arm, command, timeout):
        #the manifest on disk must already hold every arm before this one
        written = sweep.load_manifest(tmp_path / "m.json")
        seen.append(sorted(written))
        return _fake_record(arm)

    monkeypatch.setattr(sweep, "run_arm", fake_run_arm)
    sweep.main(["--manifest", str(tmp_path / "m.json"), "--encodings", "rate",
                "--algorithms", "eventprop", "--seeds", "1,2,3", "--windows", "5"])

    assert seen == [[], ["rate_eventprop_L5_s1"],
                    ["rate_eventprop_L5_s1", "rate_eventprop_L5_s2"]]
    final = json.loads((tmp_path / "m.json").read_text(encoding="utf8"))
    assert len(final) == 3


def test_dry_run_starts_nothing(tmp_path, monkeypatch):
    def explode(*args, **kwargs):
        raise AssertionError("--dry-run must not run an arm")

    monkeypatch.setattr(sweep, "run_arm", explode)
    sweep.main(["--dry-run", "--manifest", str(tmp_path / "m.json")])
    assert not (tmp_path / "m.json").exists()


def test_limit_caps_the_number_of_arms(tmp_path, monkeypatch):
    monkeypatch.setattr(sweep, "run_arm",
                        lambda arm, command, timeout: _fake_record(arm))
    sweep.main(["--limit", "3", "--manifest", str(tmp_path / "m.json")])
    assert len(json.loads((tmp_path / "m.json").read_text(encoding="utf8"))) == 3


def test_a_failed_arm_does_not_stop_the_sweep(tmp_path, monkeypatch):
    def fake_run_arm(arm, command, timeout):
        return _fake_record(arm, returncode=1 if arm.seed == 2 else 0)

    monkeypatch.setattr(sweep, "run_arm", fake_run_arm)
    sweep.main(["--manifest", str(tmp_path / "m.json"), "--encodings", "rate",
                "--algorithms", "eventprop", "--seeds", "1,2,3", "--windows", "5"])
    records = json.loads((tmp_path / "m.json").read_text(encoding="utf8"))
    assert len(records) == 3, "the sweep stopped early on a failure"
    assert [r["returncode"] for r in records] == [0, 1, 0]


def test_run_arm_records_a_timeout_rather_than_raising(monkeypatch):
    import subprocess

    def fake_subprocess_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="x", timeout=1.0)

    monkeypatch.setattr(subprocess, "run", fake_subprocess_run)
    arm = sweep.Arm(encoding="rate", algorithm="eventprop", seed=1, input_window=5)
    record = sweep.run_arm(arm, ["py"], timeout=1.0)
    assert record["returncode"] == -1
    assert "timed out" in record["error"]


@pytest.mark.parametrize("window", sweep.WINDOWS)
def test_both_sequence_lengths_are_swept(window):
    """L is a swept axis, not a fixed choice, so latency is not judged at L=5 alone."""
    assert any(a.input_window == window for a in sweep.build_grid())
