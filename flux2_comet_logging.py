from __future__ import annotations

import datetime
import logging
import os
import secrets
import subprocess
from pathlib import Path
import re
from typing import Any, Mapping, Optional

_COMET_EXPERIMENTS_BY_NAMESPACE: dict[str, dict[str, Any]] = {}
_COMET_LOGGED_PARAM_KEYS_BY_NAMESPACE: dict[str, set[str]] = {}


def _namespace_key(namespace: str) -> str:
    key = str(namespace or "").strip()
    return key if key else "default"


def experiments_for(namespace: str) -> dict[str, Any]:
    key = _namespace_key(namespace)
    cache = _COMET_EXPERIMENTS_BY_NAMESPACE.get(key)
    if cache is None:
        cache = {}
        _COMET_EXPERIMENTS_BY_NAMESPACE[key] = cache
    return cache


def logged_param_keys_for(namespace: str) -> set[str]:
    key = _namespace_key(namespace)
    cache = _COMET_LOGGED_PARAM_KEYS_BY_NAMESPACE.get(key)
    if cache is None:
        cache = set()
        _COMET_LOGGED_PARAM_KEYS_BY_NAMESPACE[key] = cache
    return cache


def clear_namespace_state(namespace: str) -> None:
    experiments_for(namespace).clear()
    logged_param_keys_for(namespace).clear()


def normalize_experiment_key(
    value: str | None,
    *,
    min_len: int = 32,
    max_len: int = 50,
    allow_empty: bool = False,
) -> str:
    if min_len <= 0:
        raise ValueError("min_len must be > 0.")
    if max_len < min_len:
        raise ValueError("max_len must be >= min_len.")

    text = "" if value is None else str(value)
    key = "".join(ch for ch in text if ch.isalnum())
    if not key:
        if allow_empty:
            return ""
        key = f"exp{secrets.token_hex(16)}"
    if len(key) < min_len:
        key = key + ("X" * (min_len - len(key)))
    if len(key) > max_len:
        key = key[:max_len]
    return key


def _resolve_git_dir(anchor_file: str) -> Optional[Path]:
    start = Path(anchor_file).resolve().parent
    for parent in (start, *start.parents):
        dot_git = parent / ".git"
        if dot_git.is_dir():
            return dot_git
        if dot_git.is_file():
            try:
                text = dot_git.read_text(encoding="utf-8", errors="ignore").strip()
            except Exception:
                continue
            prefix = "gitdir:"
            if not text.lower().startswith(prefix):
                continue
            raw_path = text[len(prefix):].strip()
            if not raw_path:
                continue
            git_dir = Path(raw_path)
            if not git_dir.is_absolute():
                git_dir = (parent / git_dir).resolve()
            if git_dir.exists():
                return git_dir
    return None


def _read_packed_ref(git_dir: Path, ref_name: str) -> Optional[str]:
    packed_refs = git_dir / "packed-refs"
    if not packed_refs.is_file():
        return None
    try:
        with packed_refs.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("^"):
                    continue
                parts = line.split(" ", 1)
                if len(parts) != 2:
                    continue
                commit_hash, ref = parts
                if ref == ref_name:
                    return commit_hash.strip()
    except Exception:
        return None
    return None


def _read_git_commit_hash(anchor_file: str) -> Optional[str]:
    git_dir = _resolve_git_dir(anchor_file)
    if git_dir is None:
        return None

    head_path = git_dir / "HEAD"
    try:
        head_text = head_path.read_text(encoding="utf-8", errors="ignore").strip()
    except Exception:
        return None
    if not head_text:
        return None

    commit_hash: Optional[str] = None
    if head_text.startswith("ref:"):
        ref_name = head_text.split(":", 1)[1].strip()
        if not ref_name:
            return None
        ref_path = git_dir / ref_name
        if ref_path.is_file():
            try:
                commit_hash = ref_path.read_text(encoding="utf-8", errors="ignore").strip()
            except Exception:
                commit_hash = None
        if not commit_hash:
            commit_hash = _read_packed_ref(git_dir, ref_name)
    else:
        commit_hash = head_text

    if not commit_hash:
        return None
    commit_hash = commit_hash.strip()
    if not re.fullmatch(r"[0-9a-fA-F]{7,40}", commit_hash):
        return None
    return commit_hash.lower()


def git_short_hash(anchor_file: str, length: int = 7) -> str:
    repo_dir = Path(anchor_file).resolve().parent
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", f"--short={max(1, int(length))}", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            cwd=str(repo_dir),
        ).strip()
        if out:
            return out
    except Exception:
        pass
    commit_hash = _read_git_commit_hash(anchor_file)
    if commit_hash:
        return commit_hash[: max(1, int(length))]
    return "nogit"


def generate_experiment_key(anchor_file: str) -> str:
    short_hash = git_short_hash(anchor_file, length=7)
    ts = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
    pid = f"{os.getpid() % 100000:05d}"
    return normalize_experiment_key(f"{short_hash}{ts}{pid}")


def generate_experiment_display_name(anchor_file: str) -> str:
    short_hash = git_short_hash(anchor_file, length=6)
    ts = datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")
    return f"{ts}-{short_hash}"


def resolve_api_key(configured_api_key: str) -> tuple[str, str]:
    configured = str(configured_api_key or "").strip()
    if configured:
        return configured, "config"
    env_value = os.getenv("COMET_API_KEY", "").strip()
    if env_value:
        return env_value, "env"
    return "", "none"


def ensure_experiment(
    *,
    namespace: str,
    logger: logging.Logger,
    component_name: str,
    enabled: bool,
    disabled: bool,
    existing_experiment: Any,
    api_key: str,
    project_name: str,
    workspace: str,
    experiment_key: str,
    display_name: str,
    persist_experiment: bool,
    params: Optional[dict[str, Any]] = None,
) -> tuple[Any, bool]:
    if not bool(enabled) or bool(disabled):
        return None, bool(disabled)
    if existing_experiment is not None:
        return existing_experiment, False

    experiment_key = str(experiment_key or "").strip()
    display_name = str(display_name or "").strip()
    workspace_value = str(workspace or "").strip()

    if bool(persist_experiment) and experiment_key:
        cached = experiments_for(namespace).get(experiment_key)
        if cached is not None:
            return cached, False

    resolved_api_key, api_key_source = resolve_api_key(api_key)
    if not resolved_api_key:
        logger.warning(
            "%s: Comet logging enabled but no API key configured; disabling Comet logging.",
            component_name,
        )
        return None, True

    try:
        from comet_ml import start
    except Exception as exc:
        logger.warning(
            "%s: could not import comet_ml; disabling Comet logging (%s).",
            component_name,
            exc,
        )
        return None, True

    try:
        params_payload = dict(params or {})
        logger.info(
            "%s: preparing Comet logging (project=%s workspace=%s experiment=%s display_name=%s persist=%s api_key_source=%s params=%s).",
            component_name,
            project_name,
            workspace_value or "<none>",
            experiment_key or "<none>",
            display_name or "<none>",
            bool(persist_experiment and experiment_key),
            api_key_source,
            params_payload,
        )

        kwargs: dict[str, Any] = {
            "api_key": resolved_api_key,
            "project_name": str(project_name or "ttr-distillation"),
            "workspace": workspace_value or None,
        }
        if experiment_key:
            kwargs["experiment_key"] = experiment_key

        experiment = start(**kwargs)

        if display_name and hasattr(experiment, "set_name"):
            try:
                experiment.set_name(display_name)
            except Exception as exc:
                logger.warning(
                    "%s: failed to set Comet display name %s (%s).",
                    component_name,
                    display_name,
                    exc,
                )

        if params_payload:
            param_key = experiment_key if experiment_key else f"runtime:{id(experiment)}"
            logged_keys = logged_param_keys_for(namespace)
            if param_key not in logged_keys:
                experiment.log_parameters(params_payload)
                logged_keys.add(param_key)

        if bool(persist_experiment) and experiment_key:
            experiments_for(namespace)[experiment_key] = experiment

        logger.info(
            "%s: Comet logging enabled (project=%s workspace=%s experiment=%s display_name=%s persist=%s).",
            component_name,
            project_name,
            workspace_value or "<none>",
            experiment_key or "<none>",
            display_name or "<none>",
            bool(persist_experiment and experiment_key),
        )
        return experiment, False
    except Exception as exc:
        logger.warning(
            "%s: failed to start Comet experiment; disabling Comet logging (%s).",
            component_name,
            exc,
        )
        return None, True


def release_experiment(
    *,
    namespace: str,
    logger: logging.Logger,
    component_name: str,
    experiment: Any,
    enabled: bool,
    persist_experiment: bool,
    experiment_key: str,
) -> None:
    if experiment is None:
        return

    experiment_key = str(experiment_key or "").strip()
    keep_open = bool(enabled and persist_experiment and experiment_key)
    if keep_open:
        return

    try:
        experiment.end()
    except Exception as exc:
        logger.warning("%s: failed to end Comet experiment cleanly: %s", component_name, exc)

    if experiment_key:
        cached = experiments_for(namespace).get(experiment_key)
        if cached is experiment:
            experiments_for(namespace).pop(experiment_key, None)
            logged_param_keys_for(namespace).discard(experiment_key)


def safe_log_metrics(
    *,
    experiment: Any,
    payload: Mapping[str, float],
    step: int,
    logger: logging.Logger,
    component_name: str,
) -> bool:
    if experiment is None or not payload:
        return False
    try:
        experiment.log_metrics(dict(payload), step=int(step))
        return True
    except Exception as exc:
        logger.warning(
            "%s: Comet metric logging failed at step %d; disabling Comet logging (%s).",
            component_name,
            int(step),
            exc,
        )
        return False


def _sanitize_artifact_name(name: str, *, fallback: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(name or "").strip())
    cleaned = cleaned.strip("._-")
    if cleaned:
        return cleaned
    fallback_cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(fallback or "").strip()).strip("._-")
    return fallback_cleaned or "checkpoint"


def safe_log_checkpoint_artifact(
    *,
    experiment: Any,
    checkpoint_path: str,
    logger: logging.Logger,
    component_name: str,
    artifact_name: str,
    step: Optional[int] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> bool:
    if experiment is None:
        return False
    path = str(checkpoint_path or "").strip()
    if not path:
        return False
    checkpoint_file = Path(path)
    if not checkpoint_file.is_file():
        logger.warning(
            "%s: Comet checkpoint artifact logging skipped; file not found: %s.",
            component_name,
            path,
        )
        return False

    try:
        from comet_ml import Artifact
    except ImportError as exc:
        logger.warning(
            "%s: could not import comet_ml Artifact for checkpoint upload (%s: %s).",
            component_name,
            type(exc).__name__,
            exc,
        )
        return False

    artifact_key = _sanitize_artifact_name(artifact_name, fallback=checkpoint_file.stem)
    metadata_payload: dict[str, Any] = dict(metadata or {})
    metadata_payload.setdefault("checkpoint_path", str(checkpoint_file))
    metadata_payload.setdefault("checkpoint_file", checkpoint_file.name)
    if step is not None:
        metadata_payload.setdefault("step", int(step))

    try:
        artifact = Artifact(
            name=artifact_key,
            artifact_type="checkpoint",
            metadata=metadata_payload,
        )
        artifact.add(
            str(checkpoint_file),
            logical_path=checkpoint_file.name,
            overwrite=True,
            copy_to_tmp=True,
        )
        experiment.log_artifact(artifact)
        logger.info(
            "%s: logged checkpoint artifact to Comet (artifact=%s path=%s step=%s).",
            component_name,
            artifact_key,
            checkpoint_file,
            int(step) if step is not None else "n/a",
        )
        return True
    except Exception as exc:
        logger.warning(
            "%s: Comet checkpoint artifact upload failed (%s: %s).",
            component_name,
            type(exc).__name__,
            exc,
        )
        return False


def should_log_step(
    *,
    step: int,
    every: int,
    total_steps: int,
    include_first_step: bool = False,
) -> bool:
    step_value = int(step)
    if step_value <= 0:
        return False

    interval = max(1, int(every))
    total_value = int(total_steps)

    if include_first_step and step_value == 1:
        return True
    if total_value > 0 and step_value == total_value:
        return True
    return step_value % interval == 0


def safe_log_parameters(
    *,
    experiment: Any,
    payload: Mapping[str, Any],
    logger: logging.Logger,
    component_name: str,
    failure_message: str,
) -> bool:
    if experiment is None or not payload:
        return False
    try:
        experiment.log_parameters(dict(payload))
        return True
    except Exception as exc:
        logger.warning("%s: %s (%s).", component_name, failure_message, exc)
        return False


def flatten_params(prefix: str, payload: Mapping[str, Any], out: dict[str, Any]) -> None:
    for key, value in payload.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flatten_params(full_key, value, out)
        elif isinstance(value, (list, tuple, set)):
            out[full_key] = ",".join(str(x) for x in value)
        else:
            out[full_key] = value


def sanitize_metric(value: Any) -> Optional[float]:
    try:
        v = float(value)
    except Exception:
        return None
    if not (v == v) or v in (float("inf"), float("-inf")):
        return None
    return v


def build_prefixed_metrics(prefix: str, metrics: Mapping[str, Any]) -> dict[str, float]:
    payload: dict[str, float] = {}
    metric_prefix = str(prefix or "").strip()
    for key, value in metrics.items():
        sanitized = sanitize_metric(value)
        if sanitized is None:
            continue
        metric_key = f"{metric_prefix}/{key}" if metric_prefix else str(key)
        payload[metric_key] = sanitized
    return payload
