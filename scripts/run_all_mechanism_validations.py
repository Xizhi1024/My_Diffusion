"""Strict one-command orchestrator for the CT->PET mechanism gates.

The runner executes the locked stage order and trusts only each stage's
``decision.json``.  A non-zero command, missing/invalid decision, mismatched
dataset contract, failed gate, or unimplemented stage blocks every dependent
stage.  It never promotes legacy checkpoints and never turns missing evidence
into a pass.

Local data-only prefix::

    pixi run mechanism-validate-local

Complete fail-closed pipeline::

    pixi run mechanism-validate-all
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PIPELINE = ROOT / "configs" / "mechanism_validation_pipeline_v1.json"
DEFAULT_CONTRACT = ROOT / "configs" / "dataset_contract_stage0a_v1.json"
DEFAULT_SUMMARY = (
    ROOT / "results" / "mechanism_validation" / "99_full_pipeline"
)
PASS = "PASS"


class PipelineSpecError(ValueError):
    """Raised when the pipeline specification is internally inconsistent."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _looks_like_raw_png_root(path: Path) -> bool:
    return any(
        (path / split / modality).is_dir()
        for split in ("train", "val", "test")
        for modality in ("ct", "pet", "label")
    )


def _select_raw_root(
    root: Path,
    explicit: Path | None,
    contract: Mapping[str, Any],
) -> Path:
    if explicit is not None:
        candidate = _resolve(root, explicit)
        if not _looks_like_raw_png_root(candidate):
            raise PipelineSpecError(
                f"--raw-root is not a CT/PET/label PNG store: {candidate}"
            )
        return candidate

    declared = str(contract.get("raw_png", {}).get("root", "") or "").strip()
    candidates: list[Path] = []
    if declared:
        candidates.append(_resolve(root, declared))
    candidates.extend((root / "main_data", root / "Data" / "data"))
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if _looks_like_raw_png_root(resolved):
            return resolved
    raise PipelineSpecError(
        "Cannot locate a CT/PET/label PNG store. Checked the contract path, "
        "main_data, and Data/data; pass --raw-root explicitly."
    )


def _nested_get(payload: Mapping[str, Any], dotted_path: str) -> Any:
    current: Any = payload
    for part in dotted_path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise KeyError(dotted_path)
        current = current[part]
    return current


def _validate_pipeline_spec(spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    if spec.get("schema_version") != 1:
        raise PipelineSpecError("pipeline schema_version must be 1")
    stages = spec.get("stages")
    if not isinstance(stages, list) or not stages:
        raise PipelineSpecError("pipeline must contain a non-empty stages list")

    profiles = spec.get("decision_profiles", {})
    if not isinstance(profiles, Mapping):
        raise PipelineSpecError("decision_profiles must be an object")
    known: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for index, raw_stage in enumerate(stages):
        if not isinstance(raw_stage, Mapping):
            raise PipelineSpecError(f"stage {index} is not an object")
        stage = dict(raw_stage)
        stage_id = str(stage.get("id", "")).strip()
        if not stage_id or stage_id in known:
            raise PipelineSpecError(f"invalid or duplicate stage id: {stage_id!r}")
        scopes = stage.get("scopes")
        if not isinstance(scopes, list) or not set(scopes).issubset({"local", "all"}):
            raise PipelineSpecError(f"{stage_id}: scopes must contain local/all")
        requires = stage.get("requires", [])
        if not isinstance(requires, list) or any(item not in known for item in requires):
            raise PipelineSpecError(
                f"{stage_id}: dependencies must reference earlier stages only"
            )
        for key in ("output", "decision_file", "pass_path", "pass_value"):
            if not isinstance(stage.get(key), str) or not stage[key]:
                raise PipelineSpecError(f"{stage_id}: missing {key}")
        implemented = stage.get("implemented")
        if not isinstance(implemented, bool):
            raise PipelineSpecError(f"{stage_id}: implemented must be boolean")
        if implemented:
            command = stage.get("command")
            if not isinstance(command, list) or not command:
                raise PipelineSpecError(
                    f"{stage_id}: implemented stage needs a command list"
                )
            if any(not isinstance(token, str) for token in command):
                raise PipelineSpecError(f"{stage_id}: command tokens must be strings")
        profile_name = stage.get("decision_profile")
        profile_assertions: Mapping[str, Any] = {}
        if profile_name is not None:
            profile = profiles.get(profile_name)
            if not isinstance(profile, Mapping):
                raise PipelineSpecError(
                    f"{stage_id}: unknown decision_profile {profile_name!r}"
                )
            profile_assertions = profile.get("required_assertions", {})
            if not isinstance(profile_assertions, Mapping):
                raise PipelineSpecError(
                    f"{stage_id}: profile required_assertions must be an object"
                )
        stage_assertions = stage.get("required_assertions", {})
        if not isinstance(stage_assertions, Mapping):
            raise PipelineSpecError(
                f"{stage_id}: required_assertions must be an object"
            )
        stage["_required_assertions"] = {
            **dict(profile_assertions),
            **dict(stage_assertions),
        }
        known.add(stage_id)
        normalized.append(stage)
    return normalized


def _selected_stages(
    stages: Sequence[dict[str, Any]],
    scope: str,
    from_stage: str | None,
    through_stage: str | None,
) -> list[dict[str, Any]]:
    stage_ids = [stage["id"] for stage in stages]
    if from_stage is not None and from_stage not in stage_ids:
        raise PipelineSpecError(f"unknown --from-stage: {from_stage}")
    if through_stage is not None and through_stage not in stage_ids:
        raise PipelineSpecError(f"unknown --through-stage: {through_stage}")
    start = 0 if from_stage is None else stage_ids.index(from_stage)
    stop = len(stages) if through_stage is None else stage_ids.index(through_stage) + 1
    if start >= stop:
        raise PipelineSpecError("--from-stage must precede --through-stage")
    return [
        stage
        for index, stage in enumerate(stages)
        if start <= index < stop and scope in stage["scopes"]
    ]


def _format_command(
    stage: Mapping[str, Any],
    *,
    root: Path,
    contract: Path,
    raw_root: Path,
    cache_dir: Path,
) -> tuple[list[str], Path]:
    output = _resolve(root, str(stage["output"]))
    context = {
        "python": sys.executable,
        "root": str(root),
        "contract": str(contract),
        "raw_root": str(raw_root),
        "cache_dir": str(cache_dir),
        "output": str(output),
    }
    try:
        command = [token.format_map(context) for token in stage["command"]]
    except KeyError as exc:
        raise PipelineSpecError(
            f"{stage['id']}: unknown command placeholder {exc}"
        ) from exc
    return command, output


def _decision_passes(
    stage: Mapping[str, Any],
    payload: Mapping[str, Any],
    expected_contract_sha256: str,
) -> tuple[bool, list[str]]:
    errors: list[str] = []
    try:
        observed = _nested_get(payload, str(stage["pass_path"]))
    except KeyError:
        observed = None
        errors.append(f"missing pass field: {stage['pass_path']}")
    if observed != stage["pass_value"]:
        errors.append(
            f"gate value {stage['pass_path']}={observed!r}, "
            f"expected {stage['pass_value']!r}"
        )
    contract_path = stage.get("contract_path")
    if contract_path:
        try:
            observed_contract = _nested_get(payload, str(contract_path))
        except KeyError:
            errors.append(f"missing dataset-contract field: {contract_path}")
        else:
            if observed_contract != expected_contract_sha256:
                errors.append(
                    "dataset-contract mismatch: "
                    f"{observed_contract!r} != {expected_contract_sha256!r}"
                )
    for assertion_path, expected_value in stage.get(
        "_required_assertions", {}
    ).items():
        try:
            observed_value = _nested_get(payload, str(assertion_path))
        except KeyError:
            errors.append(f"missing required assertion: {assertion_path}")
        else:
            if observed_value != expected_value:
                errors.append(
                    f"required assertion {assertion_path}={observed_value!r}, "
                    f"expected {expected_value!r}"
                )
    return not errors, errors


def _run_command(
    command: Sequence[str],
    *,
    cwd: Path,
    log_path: Path,
) -> tuple[int, float]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment["PYTHONUNBUFFERED"] = "1"
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8", newline="") as log:
        process = subprocess.Popen(
            list(command),
            cwd=str(cwd),
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
        except KeyboardInterrupt:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
            raise
        return process.wait(), time.monotonic() - started


def _mirror_stage_decision(
    run_dir: Path,
    stage_id: str,
    payload: Mapping[str, Any],
) -> None:
    _write_json(run_dir / stage_id / "decision.json", payload)


def _verified_external_prerequisites(
    *,
    all_stages: Sequence[Mapping[str, Any]],
    selected_stages: Sequence[Mapping[str, Any]],
    root: Path,
    expected_contract_sha256: str,
) -> tuple[set[str], list[dict[str, Any]], str | None]:
    """Verify canonical gates omitted by ``--from-stage``.

    A resumed run never assumes that an earlier command passed.  It loads the
    canonical decision, checks the locked contract and decision-profile
    assertions, and only then treats that dependency as satisfied.
    """

    selected_ids = {str(stage["id"]) for stage in selected_stages}
    by_id = {str(stage["id"]): stage for stage in all_stages}
    required: set[str] = set()

    def visit(stage_id: str) -> None:
        for dependency in by_id[stage_id].get("requires", []):
            dependency = str(dependency)
            if dependency in required:
                continue
            required.add(dependency)
            visit(dependency)

    for stage in selected_stages:
        visit(str(stage["id"]))
    external_ids = required - selected_ids
    verified: set[str] = set()
    records: list[dict[str, Any]] = []
    blocker: str | None = None
    for stage in all_stages:
        stage_id = str(stage["id"])
        if stage_id not in external_ids:
            continue
        decision_path = (
            _resolve(root, str(stage["output"]))
            / str(stage["decision_file"])
        )
        errors: list[str] = []
        payload: Mapping[str, Any] | None = None
        if not decision_path.is_file():
            errors.append(f"missing canonical decision: {decision_path}")
        else:
            try:
                loaded = _load_json(decision_path)
            except (OSError, json.JSONDecodeError) as exc:
                errors.append(
                    f"invalid canonical decision: {type(exc).__name__}: {exc}"
                )
            else:
                if not isinstance(loaded, Mapping):
                    errors.append("canonical decision root is not an object")
                else:
                    payload = loaded
                    passed, decision_errors = _decision_passes(
                        stage,
                        loaded,
                        expected_contract_sha256,
                    )
                    if not passed:
                        errors.extend(decision_errors)
        if errors and blocker is None:
            blocker = stage_id
        elif not errors:
            verified.add(stage_id)
        records.append(
            {
                "stage_id": stage_id,
                "status": PASS if not errors else "FAIL",
                "decision_path": str(decision_path),
                "decision_sha256": (
                    _sha256_file(decision_path) if payload is not None else None
                ),
                "errors": errors,
            }
        )
    return verified, records, blocker


def _base_stage_record(stage: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "stage_id": stage["id"],
        "title": stage.get("title"),
        "hypothesis": stage.get("hypothesis"),
        "requires": list(stage.get("requires", [])),
        "created_at_utc": _utc_now(),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = args.root.resolve()
    pipeline_path = _resolve(root, args.pipeline)
    contract_path = _resolve(root, args.contract)
    cache_dir = _resolve(root, args.cache_dir)
    summary_dir = _resolve(root, args.summary_dir)

    spec = _load_json(pipeline_path)
    if not isinstance(spec, Mapping):
        raise PipelineSpecError("pipeline JSON root must be an object")
    stages = _validate_pipeline_spec(spec)
    selected = _selected_stages(
        stages,
        args.scope,
        args.from_stage,
        args.through_stage,
    )
    if not selected:
        raise PipelineSpecError("stage selection is empty")

    contract = _load_json(contract_path)
    if not isinstance(contract, Mapping) or not isinstance(
        contract.get("contract_sha256"), str
    ):
        raise PipelineSpecError("dataset contract lacks contract_sha256")
    contract_sha256 = str(contract["contract_sha256"])
    raw_root = _select_raw_root(root, args.raw_root, contract)
    (
        external_prerequisites,
        external_prerequisite_records,
        external_prerequisite_blocker,
    ) = _verified_external_prerequisites(
        all_stages=stages,
        selected_stages=selected,
        root=root,
        expected_contract_sha256=contract_sha256,
    )

    run_id = args.run_id or datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S.%fZ"
    )
    run_dir = summary_dir / "runs" / run_id
    if run_dir.exists() and any(run_dir.iterdir()):
        raise PipelineSpecError(f"run directory already exists and is non-empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    pipeline_sha256 = _sha256_file(pipeline_path)
    unimplemented = [stage for stage in selected if not stage["implemented"]]
    preflight = {
        "schema_version": 1,
        "pipeline_id": spec.get("pipeline_id"),
        "pipeline_sha256": pipeline_sha256,
        "scope": args.scope,
        "selected_stage_ids": [stage["id"] for stage in selected],
        "dataset_contract_sha256": contract_sha256,
        "selected_raw_root": str(raw_root),
        "unimplemented_stage_ids": [stage["id"] for stage in unimplemented],
        "external_prerequisites": external_prerequisite_records,
        "commands_executed": False,
        "created_at_utc": _utc_now(),
    }
    _write_json(run_dir / "preflight.json", preflight)

    if args.dry_run:
        stage_records = []
        for stage in selected:
            command = None
            output = _resolve(root, str(stage["output"]))
            if stage["implemented"]:
                command, output = _format_command(
                    stage,
                    root=root,
                    contract=contract_path,
                    raw_root=raw_root,
                    cache_dir=cache_dir,
                )
            record = {
                **_base_stage_record(stage),
                "status": "DRY_RUN",
                "implemented": stage["implemented"],
                "command": command,
                "output": str(output),
                "expected_script": stage.get("expected_script"),
            }
            _mirror_stage_decision(run_dir, stage["id"], record)
            stage_records.append(record)
        overall = _overall_decision(
            args=args,
            spec=spec,
            pipeline_path=pipeline_path,
            pipeline_sha256=pipeline_sha256,
            contract_sha256=contract_sha256,
            run_id=run_id,
            run_dir=run_dir,
            stage_records=stage_records,
            decision="DRY_RUN",
            commands_executed=False,
        )
        _write_json(run_dir / "decision.json", overall)
        _write_json(summary_dir / "decision.json", overall)
        return overall

    if unimplemented and not args.allow_partial:
        stage_records = []
        unimplemented_ids = {stage["id"] for stage in unimplemented}
        for stage in selected:
            status = (
                "NOT_IMPLEMENTED"
                if stage["id"] in unimplemented_ids
                else "BLOCKED_BY_INCOMPLETE_PIPELINE_PREFLIGHT"
            )
            record = {
                **_base_stage_record(stage),
                "status": status,
                "canonical_decision_used": False,
                "expected_script": stage.get("expected_script"),
                "reason": (
                    "The formal validator has not been implemented."
                    if status == "NOT_IMPLEMENTED"
                    else "No stage was run because --all is fail-closed until the selected pipeline is complete."
                ),
            }
            _mirror_stage_decision(run_dir, stage["id"], record)
            stage_records.append(record)
        overall = _overall_decision(
            args=args,
            spec=spec,
            pipeline_path=pipeline_path,
            pipeline_sha256=pipeline_sha256,
            contract_sha256=contract_sha256,
            run_id=run_id,
            run_dir=run_dir,
            stage_records=stage_records,
            decision="NOT_IMPLEMENTED",
            commands_executed=False,
        )
        _write_json(run_dir / "decision.json", overall)
        _write_json(summary_dir / "decision.json", overall)
        return overall

    stage_records: list[dict[str, Any]] = []
    passed_stage_ids: set[str] = set(external_prerequisites)
    upstream_blocker: str | None = external_prerequisite_blocker
    commands_executed = False

    for stage in selected:
        base = _base_stage_record(stage)
        missing_dependencies = [
            required
            for required in stage.get("requires", [])
            if required not in passed_stage_ids
        ]
        if upstream_blocker is not None or missing_dependencies:
            record = {
                **base,
                "status": "BLOCKED_UPSTREAM",
                "canonical_decision_used": False,
                "blocker": upstream_blocker,
                "missing_passed_dependencies": missing_dependencies,
            }
            _mirror_stage_decision(run_dir, stage["id"], record)
            stage_records.append(record)
            continue

        if not stage["implemented"]:
            record = {
                **base,
                "status": "NOT_IMPLEMENTED",
                "canonical_decision_used": False,
                "expected_script": stage.get("expected_script"),
            }
            upstream_blocker = stage["id"]
            _mirror_stage_decision(run_dir, stage["id"], record)
            stage_records.append(record)
            continue

        command, output = _format_command(
            stage,
            root=root,
            contract=contract_path,
            raw_root=raw_root,
            cache_dir=cache_dir,
        )
        decision_path = output / str(stage["decision_file"])
        log_path = run_dir / stage["id"] / "command.log"
        print(f"\n=== {stage['id']}: {stage.get('title', '')} ===", flush=True)
        print(subprocess.list2cmdline(command), flush=True)
        commands_executed = True
        exit_code, duration = _run_command(command, cwd=root, log_path=log_path)

        errors: list[str] = []
        canonical_payload: Mapping[str, Any] | None = None
        if not decision_path.is_file():
            errors.append(f"missing decision file: {decision_path}")
        else:
            try:
                loaded = _load_json(decision_path)
            except (OSError, json.JSONDecodeError) as exc:
                errors.append(f"invalid decision JSON: {type(exc).__name__}: {exc}")
            else:
                if not isinstance(loaded, Mapping):
                    errors.append("decision JSON root is not an object")
                else:
                    canonical_payload = loaded
                    passed, decision_errors = _decision_passes(
                        stage,
                        loaded,
                        contract_sha256,
                    )
                    if not passed:
                        errors.extend(decision_errors)
        if exit_code != 0:
            errors.append(f"stage command exited with code {exit_code}")

        status = PASS if not errors else "FAIL"
        record = {
            **base,
            "status": status,
            "canonical_decision_used": canonical_payload is not None,
            "canonical_decision_path": str(decision_path),
            "canonical_decision_sha256": (
                _sha256_file(decision_path) if canonical_payload is not None else None
            ),
            "command": command,
            "command_exit_code": exit_code,
            "duration_seconds": duration,
            "log_path": str(log_path),
            "errors": errors,
            "canonical_decision": canonical_payload,
        }
        if status == PASS:
            passed_stage_ids.add(stage["id"])
        else:
            upstream_blocker = stage["id"]
        _mirror_stage_decision(run_dir, stage["id"], record)
        stage_records.append(record)

    statuses = [record["status"] for record in stage_records]
    if statuses and all(status == PASS for status in statuses):
        final_decision = PASS
    elif "NOT_IMPLEMENTED" in statuses:
        final_decision = "NOT_IMPLEMENTED"
    else:
        final_decision = "FAIL"
    overall = _overall_decision(
        args=args,
        spec=spec,
        pipeline_path=pipeline_path,
        pipeline_sha256=pipeline_sha256,
        contract_sha256=contract_sha256,
        run_id=run_id,
        run_dir=run_dir,
        stage_records=stage_records,
        decision=final_decision,
        commands_executed=commands_executed,
    )
    _write_json(run_dir / "decision.json", overall)
    _write_json(summary_dir / "decision.json", overall)
    return overall


def _overall_decision(
    *,
    args: argparse.Namespace,
    spec: Mapping[str, Any],
    pipeline_path: Path,
    pipeline_sha256: str,
    contract_sha256: str,
    run_id: str,
    run_dir: Path,
    stage_records: Sequence[Mapping[str, Any]],
    decision: str,
    commands_executed: bool,
) -> dict[str, Any]:
    hypothesis_status: dict[str, str] = {}
    for hypothesis in ("H1", "H2", "H3", "H4", "H5", "H6"):
        matching = [
            record
            for record in stage_records
            if record.get("hypothesis") == hypothesis
        ]
        hypothesis_status[hypothesis] = (
            str(matching[-1]["status"]) if matching else "NOT_SELECTED"
        )
    all_hypotheses_validated = bool(
        args.scope == "all"
        and all(hypothesis_status[key] == PASS for key in hypothesis_status)
    )
    return {
        "schema_version": 1,
        "pipeline_id": spec.get("pipeline_id"),
        "run_id": run_id,
        "scope": args.scope,
        "decision": decision,
        "all_hypotheses_validated": all_hypotheses_validated,
        "model_mechanism_claims_allowed": all_hypotheses_validated,
        "formal_final_performance_claims_allowed": bool(
            all_hypotheses_validated and hypothesis_status["H6"] == PASS
        ),
        "dataset_contract_sha256": contract_sha256,
        "pipeline_spec": str(pipeline_path),
        "pipeline_sha256": pipeline_sha256,
        "policy": spec.get("policy"),
        "commands_executed": commands_executed,
        "hypothesis_status": hypothesis_status,
        "stage_status": {
            str(record["stage_id"]): str(record["status"])
            for record in stage_records
        },
        "stage_decisions": [
            str(run_dir / str(record["stage_id"]) / "decision.json")
            for record in stage_records
        ],
        "run_directory": str(run_dir),
        "created_at_utc": _utc_now(),
        "stop_rule": (
            "All selected gates passed."
            if decision == PASS
            else "Do not promote blocked hypotheses or dependent modules."
        ),
    }


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--pipeline", type=Path, default=DEFAULT_PIPELINE)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=None,
        help=(
            "Physical PNG root override. By default the runner selects the "
            "locked contract path, main_data, or Data/data by directory shape; "
            "Stage 0A then verifies the complete locked byte fingerprint."
        ),
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("cache/tensors_main"))
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--scope", choices=("local", "all"), default="all")
    parser.add_argument("--from-stage", default=None)
    parser.add_argument("--through-stage", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help=(
            "Run the implemented prefix and stop at the first missing validator. "
            "Without this flag, an incomplete selected pipeline fails in preflight."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    args.root = args.root.resolve()
    return args


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        decision = run(args)
    except KeyboardInterrupt:
        print("\nInterrupted; no gate was promoted.", file=sys.stderr, flush=True)
        return 130
    except Exception as exc:
        failure = {
            "schema_version": 1,
            "pipeline_id": "unknown",
            "decision": "ERROR",
            "all_hypotheses_validated": False,
            "model_mechanism_claims_allowed": False,
            "formal_final_performance_claims_allowed": False,
            "fatal_error": f"{type(exc).__name__}: {exc}",
            "python": sys.version,
            "platform": platform.platform(),
            "created_at_utc": _utc_now(),
            "stop_rule": "Fix the runner/specification error before any mechanism claim.",
        }
        summary_dir = _resolve(args.root, args.summary_dir)
        _write_json(summary_dir / "decision.json", failure)
        print(json.dumps(failure, ensure_ascii=False, indent=2), flush=True)
        return 2
    print(json.dumps(decision, ensure_ascii=False, indent=2), flush=True)
    return 0 if decision["decision"] in {PASS, "DRY_RUN"} else 3


if __name__ == "__main__":
    raise SystemExit(main())
