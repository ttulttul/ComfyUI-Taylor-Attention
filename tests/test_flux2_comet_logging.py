from __future__ import annotations

import logging
import sys
import types
from pathlib import Path

import flux2_comet_logging


def test_ensure_experiment_reuses_cached_experiment_and_logs_params_once(monkeypatch):
    start_calls = []
    set_name_calls = []
    params_calls = []

    class _FakeExperiment:
        def set_name(self, name):
            set_name_calls.append(str(name))

        def log_parameters(self, params):
            params_calls.append(dict(params))

        def log_metrics(self, metrics, step=None):
            del metrics, step
            return None

        def end(self):
            return None

    def _fake_start(api_key=None, project_name=None, workspace=None, experiment_key=None):
        start_calls.append((api_key, project_name, workspace, experiment_key))
        return _FakeExperiment()

    fake_comet = types.ModuleType("comet_ml")
    fake_comet.start = _fake_start
    monkeypatch.setitem(sys.modules, "comet_ml", fake_comet)

    namespace = "test_shared_comet"
    flux2_comet_logging.clear_namespace_state(namespace)

    logger = logging.getLogger("test_flux2_comet_logging")
    exp_a, disabled_a = flux2_comet_logging.ensure_experiment(
        namespace=namespace,
        logger=logger,
        component_name="TestComet",
        enabled=True,
        disabled=False,
        existing_experiment=None,
        api_key="api-key",
        project_name="proj",
        workspace="ws",
        experiment_key="exp-key",
        display_name="display-name",
        persist_experiment=True,
        params={"alpha": 1.0},
    )
    exp_b, disabled_b = flux2_comet_logging.ensure_experiment(
        namespace=namespace,
        logger=logger,
        component_name="TestComet",
        enabled=True,
        disabled=False,
        existing_experiment=None,
        api_key="api-key",
        project_name="proj",
        workspace="ws",
        experiment_key="exp-key",
        display_name="display-name",
        persist_experiment=True,
        params={"alpha": 1.0},
    )

    assert disabled_a is False
    assert disabled_b is False
    assert exp_a is not None
    assert exp_b is exp_a
    assert start_calls == [("api-key", "proj", "ws", "exp-key")]
    assert set_name_calls == ["display-name"]
    assert params_calls == [{"alpha": 1.0}]


def test_release_experiment_ends_and_clears_cache_when_not_persistent():
    end_calls = []

    class _FakeExperiment:
        def end(self):
            end_calls.append(True)

    namespace = "test_release_comet"
    flux2_comet_logging.clear_namespace_state(namespace)
    exp = _FakeExperiment()
    flux2_comet_logging.experiments_for(namespace)["exp-key"] = exp
    flux2_comet_logging.logged_param_keys_for(namespace).add("exp-key")

    flux2_comet_logging.release_experiment(
        namespace=namespace,
        logger=logging.getLogger("test_flux2_comet_logging"),
        component_name="TestComet",
        experiment=exp,
        enabled=True,
        persist_experiment=False,
        experiment_key="exp-key",
    )

    assert end_calls == [True]
    assert "exp-key" not in flux2_comet_logging.experiments_for(namespace)
    assert "exp-key" not in flux2_comet_logging.logged_param_keys_for(namespace)


def test_flatten_params_and_build_prefixed_metrics():
    flattened: dict[str, object] = {}
    flux2_comet_logging.flatten_params(
        "training",
        {
            "optimizer": {"lr": 1e-4},
            "layers": [0, 1, 2],
        },
        flattened,
    )
    assert flattened == {
        "training.optimizer.lr": 1e-4,
        "training.layers": "0,1,2",
    }

    payload = flux2_comet_logging.build_prefixed_metrics(
        "flux2ttr_controller",
        {"loss": 1.2, "bad": "text", "nan_value": float("nan")},
    )
    assert payload == {"flux2ttr_controller/loss": 1.2}


def test_safe_log_metrics_returns_false_on_experiment_error():
    class _FakeExperiment:
        def log_metrics(self, metrics, step=None):
            del metrics, step
            raise RuntimeError("boom")

    ok = flux2_comet_logging.safe_log_metrics(
        experiment=_FakeExperiment(),
        payload={"m": 1.0},
        step=3,
        logger=logging.getLogger("test_flux2_comet_logging"),
        component_name="TestComet",
    )
    assert ok is False


def test_safe_log_checkpoint_artifact_uploads_artifact(monkeypatch, tmp_path: Path):
    init_calls = []
    add_calls = []
    log_calls = []

    class _FakeArtifact:
        def __init__(self, name, artifact_type=None, version=None, aliases=None, metadata=None, version_tags=None):
            del version, aliases, version_tags
            init_calls.append((name, artifact_type, dict(metadata or {})))

        def add(self, local_path_or_data, logical_path=None, overwrite=False, copy_to_tmp=True, metadata=None):
            add_calls.append((local_path_or_data, logical_path, overwrite, copy_to_tmp, metadata))

    class _FakeExperiment:
        def log_artifact(self, artifact):
            log_calls.append(artifact)
            return {"logged": True}

    fake_comet = types.ModuleType("comet_ml")
    fake_comet.Artifact = _FakeArtifact
    monkeypatch.setitem(sys.modules, "comet_ml", fake_comet)

    ckpt = tmp_path / "controller.pt"
    ckpt.write_bytes(b"checkpoint-bytes")
    ok = flux2_comet_logging.safe_log_checkpoint_artifact(
        experiment=_FakeExperiment(),
        checkpoint_path=str(ckpt),
        logger=logging.getLogger("test_flux2_comet_logging"),
        component_name="TestComet",
        artifact_name="flux2ttr-controller-checkpoint",
        step=11,
        metadata={"save_reason": "periodic"},
    )

    assert ok is True
    assert len(init_calls) == 1
    assert init_calls[0][0] == "flux2ttr-controller-checkpoint"
    assert init_calls[0][1] == "checkpoint"
    assert init_calls[0][2]["save_reason"] == "periodic"
    assert init_calls[0][2]["checkpoint_file"] == "controller.pt"
    assert init_calls[0][2]["step"] == 11
    assert len(add_calls) == 1
    assert add_calls[0][0] == str(ckpt)
    assert add_calls[0][1] == "controller.pt"
    assert add_calls[0][2] is True
    assert add_calls[0][3] is True
    assert len(log_calls) == 1


def test_safe_log_checkpoint_artifact_returns_false_on_log_failure(monkeypatch, tmp_path: Path):
    class _FakeArtifact:
        def __init__(self, name, artifact_type=None, version=None, aliases=None, metadata=None, version_tags=None):
            del name, artifact_type, version, aliases, metadata, version_tags

        def add(self, local_path_or_data, logical_path=None, overwrite=False, copy_to_tmp=True, metadata=None):
            del local_path_or_data, logical_path, overwrite, copy_to_tmp, metadata

    class _FakeExperiment:
        def log_artifact(self, artifact):
            del artifact
            raise RuntimeError("upload failed")

    fake_comet = types.ModuleType("comet_ml")
    fake_comet.Artifact = _FakeArtifact
    monkeypatch.setitem(sys.modules, "comet_ml", fake_comet)

    ckpt = tmp_path / "runtime.pt"
    ckpt.write_bytes(b"checkpoint-bytes")
    ok = flux2_comet_logging.safe_log_checkpoint_artifact(
        experiment=_FakeExperiment(),
        checkpoint_path=str(ckpt),
        logger=logging.getLogger("test_flux2_comet_logging"),
        component_name="TestComet",
        artifact_name="flux2ttr-runtime-checkpoint",
        step=5,
    )
    assert ok is False


def test_should_log_step_can_force_first_step():
    assert flux2_comet_logging.should_log_step(
        step=1,
        every=10,
        total_steps=100,
        include_first_step=True,
    )
    assert not flux2_comet_logging.should_log_step(
        step=2,
        every=10,
        total_steps=100,
        include_first_step=True,
    )
    assert flux2_comet_logging.should_log_step(
        step=10,
        every=10,
        total_steps=100,
        include_first_step=True,
    )


def test_should_log_step_uses_interval_and_final_step_by_default():
    assert not flux2_comet_logging.should_log_step(
        step=1,
        every=10,
        total_steps=100,
        include_first_step=False,
    )
    assert flux2_comet_logging.should_log_step(
        step=10,
        every=10,
        total_steps=100,
        include_first_step=False,
    )
    assert flux2_comet_logging.should_log_step(
        step=7,
        every=10,
        total_steps=7,
        include_first_step=False,
    )


def test_normalize_experiment_key_enforces_comet_constraints():
    key = flux2_comet_logging.normalize_experiment_key("nogit-20260211-104443")
    assert key.isalnum()
    assert len(key) >= 32
    assert len(key) <= 50


def test_generate_experiment_key_is_alnum_and_valid_length():
    key = flux2_comet_logging.generate_experiment_key(__file__)
    assert key.isalnum()
    assert 32 <= len(key) <= 50


def test_git_short_hash_reads_git_metadata_when_git_cli_fails(monkeypatch, tmp_path: Path):
    repo = tmp_path / "repo"
    git_dir = repo / ".git"
    ref_dir = git_dir / "refs" / "heads"
    ref_dir.mkdir(parents=True, exist_ok=True)
    commit = "0123456789abcdef0123456789abcdef01234567"
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (ref_dir / "main").write_text(f"{commit}\n", encoding="utf-8")

    anchor = repo / "flux2_comet_logging.py"
    anchor.write_text("# test anchor\n", encoding="utf-8")

    def _boom(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("git unavailable")

    monkeypatch.setattr("subprocess.check_output", _boom)

    out = flux2_comet_logging.git_short_hash(str(anchor), length=8)
    assert out == commit[:8]
