from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 only.
    import tomli as tomllib  # type: ignore[no-redef]

from utils.model_factory import normalize_backbone_name, normalize_model_name, validate_model_backbone


DEFAULT_ARCHITECTURES = ("deeplabv3plus", "unetplusplus", "unet")
DEFAULT_BACKBONES = ("efficientnet-b3", "inceptionv4", "densenet169")
DEFAULT_DATASETS = ("tu", "drac")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run training and visualization experiments sequentially."
    )
    parser.add_argument("--config", default="configs/default.toml")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--architectures", nargs="+", default=list(DEFAULT_ARCHITECTURES))
    parser.add_argument("--backbones", nargs="+", default=list(DEFAULT_BACKBONES))
    parser.add_argument("--model-root", default=None)
    parser.add_argument("--k-folds", type=int, default=None)
    parser.add_argument(
        "--name-template",
        default="{dataset}_{model}_{backbone_token}",
        help="Experiment name template. Fields: dataset, model, backbone, backbone_token.",
    )
    parser.add_argument(
        "--skip-invalid",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip architecture/backbone combinations rejected by the model factory.",
    )
    parser.add_argument(
        "--skip-completed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip experiments whose visualization metrics already exist.",
    )
    parser.add_argument(
        "--continue-on-error",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Continue with the next experiment after a train or visualization failure.",
    )
    parser.add_argument(
        "--nearby-filter",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override visualization nearby-filter setting from the config.",
    )
    parser.add_argument(
        "--train-arg",
        action="append",
        default=[],
        help="Extra argument token for main.py. Repeat for each token, e.g. --train-arg=--epochs --train-arg=50.",
    )
    parser.add_argument(
        "--visualize-arg",
        action="append",
        default=[],
        help="Extra argument token for visualize_nearby.py. Repeat for each token.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        return tomllib.load(handle)


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def backbone_token(backbone: str) -> str:
    return normalize_backbone_name(backbone).replace("-", "")


def dataset_token(dataset: str) -> str:
    return dataset.lower().replace("dataset_", "")


def experiment_name(template: str, dataset: str, model: str, backbone: str) -> str:
    normalized_model = normalize_model_name(model)
    normalized_backbone = normalize_backbone_name(backbone)
    return template.format(
        dataset=dataset_token(dataset),
        model=slug(normalized_model),
        backbone=slug(normalized_backbone),
        backbone_token=backbone_token(normalized_backbone),
    )


def build_plan(args: argparse.Namespace) -> tuple[list[dict[str, str]], list[tuple[str, str, str, str]]]:
    plan: list[dict[str, str]] = []
    invalid: list[tuple[str, str, str, str]] = []
    datasets = [dataset_token(dataset) for dataset in args.datasets]
    architectures = [normalize_model_name(model) for model in args.architectures]
    backbones = [normalize_backbone_name(backbone) for backbone in args.backbones]

    for dataset in datasets:
        for model in architectures:
            for backbone in backbones:
                try:
                    validate_model_backbone(model, backbone)
                except ValueError as exc:
                    invalid.append((dataset, model, backbone, str(exc)))
                    if args.skip_invalid:
                        continue
                    raise
                plan.append({
                    "dataset": dataset,
                    "model": model,
                    "backbone": backbone,
                    "experiment_name": experiment_name(args.name_template, dataset, model, backbone),
                })
    return plan, invalid


def get_model_root(args: argparse.Namespace, config: dict[str, Any]) -> Path:
    if args.model_root:
        return Path(args.model_root)
    train_config = config.get("train", {})
    if isinstance(train_config, dict) and train_config.get("model_root"):
        return Path(str(train_config["model_root"]))
    return Path("model")


def get_k_folds(args: argparse.Namespace, config: dict[str, Any]) -> int:
    if args.k_folds is not None:
        return args.k_folds
    train_config = config.get("train", {})
    if isinstance(train_config, dict) and train_config.get("k_folds"):
        return int(train_config["k_folds"])
    return 5


def get_nearby_filter(args: argparse.Namespace, config: dict[str, Any]) -> bool:
    if args.nearby_filter is not None:
        return bool(args.nearby_filter)
    visualize_config = config.get("visualize", {})
    if isinstance(visualize_config, dict):
        return bool(visualize_config.get("apply_nearby_filter", False))
    return False


def visualization_metrics_path(model_root: Path, experiment_name: str, nearby_filter: bool) -> Path:
    output_dir = "outputs_nearby" if nearby_filter else "outputs"
    return model_root / experiment_name / output_dir / "test_metrics.json"


def weights_complete(model_root: Path, experiment_name: str, k_folds: int) -> bool:
    weights_dir = model_root / experiment_name / "weights"
    return all((weights_dir / f"{fold}.pth").exists() for fold in range(1, k_folds + 1))


def command_for_train(
    args: argparse.Namespace,
    experiment: dict[str, str],
) -> list[str]:
    command = [
        args.python,
        "main.py",
        "--config",
        args.config,
        "--dataset",
        experiment["dataset"],
        "--model-name",
        experiment["model"],
        "--backbone",
        experiment["backbone"],
        "--experiment-name",
        experiment["experiment_name"],
    ]
    if args.model_root:
        command.extend(["--model-root", args.model_root])
    if args.k_folds is not None:
        command.extend(["--k-folds", str(args.k_folds)])
    command.extend(args.train_arg)
    return command


def command_for_visualize(
    args: argparse.Namespace,
    experiment: dict[str, str],
) -> list[str]:
    command = [
        args.python,
        "visualize_nearby.py",
        "--config",
        args.config,
        "--dataset",
        experiment["dataset"],
        "--model-name",
        experiment["model"],
        "--backbone",
        experiment["backbone"],
        "--experiment-name",
        experiment["experiment_name"],
    ]
    if args.model_root:
        command.extend(["--model-root", args.model_root])
    if args.k_folds is not None:
        command.extend(["--k-folds", str(args.k_folds)])
    if args.nearby_filter is not None:
        command.append("--nearby-filter" if args.nearby_filter else "--no-nearby-filter")
    command.extend(args.visualize_arg)
    return command


def write_state(state_path: Path, state: dict[str, Any]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def run_command(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("$ " + " ".join(command), flush=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write("$ " + " ".join(command) + "\n")
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
        return process.wait()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    model_root = get_model_root(args, config)
    k_folds = get_k_folds(args, config)
    nearby_filter = get_nearby_filter(args, config)
    logs_dir = model_root / "experiment_queue_logs"
    state_path = model_root / "experiment_queue_state.json"

    try:
        plan, invalid = build_plan(args)
    except ValueError as exc:
        print(f"Invalid experiment plan: {exc}", file=sys.stderr)
        return 2

    total_requested = len(args.datasets) * len(args.architectures) * len(args.backbones)
    print(f"Requested experiments: {total_requested}")
    print(f"Runnable experiments: {len(plan)}")
    if invalid:
        print(f"Skipped invalid combinations: {len(invalid)}")
        for dataset, model, backbone, reason in invalid:
            print(f"  - {dataset} / {model} / {backbone}: {reason}")

    for index, experiment in enumerate(plan, start=1):
        train_command = command_for_train(args, experiment)
        visualize_command = command_for_visualize(args, experiment)
        print(f"\n[{index}/{len(plan)}] {experiment['experiment_name']}")
        print("Train: " + " ".join(train_command))
        print("Visualize: " + " ".join(visualize_command))

    if args.dry_run:
        return 0

    state: dict[str, Any] = {
        "config": args.config,
        "model_root": str(model_root),
        "k_folds": k_folds,
        "nearby_filter": nearby_filter,
        "requested_experiments": total_requested,
        "runnable_experiments": len(plan),
        "experiments": [],
    }

    for index, experiment in enumerate(plan, start=1):
        name = experiment["experiment_name"]
        metrics_path = visualization_metrics_path(model_root, name, nearby_filter)
        entry: dict[str, Any] = {
            "index": index,
            **experiment,
            "status": "pending",
            "metrics_path": str(metrics_path),
        }
        state["experiments"].append(entry)
        write_state(state_path, state)

        print(f"\n=== [{index}/{len(plan)}] {name} ===", flush=True)
        if args.skip_completed and metrics_path.exists():
            print(f"Skipping completed experiment: {metrics_path}")
            entry["status"] = "skipped_completed"
            write_state(state_path, state)
            continue

        if args.skip_completed and weights_complete(model_root, name, k_folds):
            print("Training weights already exist; running visualization only.")
        else:
            entry["status"] = "training"
            write_state(state_path, state)
            train_status = run_command(command_for_train(args, experiment), logs_dir / f"{name}_train.log")
            if train_status != 0:
                entry["status"] = "train_failed"
                entry["return_code"] = train_status
                write_state(state_path, state)
                if not args.continue_on_error:
                    return train_status
                continue

        entry["status"] = "visualizing"
        write_state(state_path, state)
        visualize_status = run_command(
            command_for_visualize(args, experiment),
            logs_dir / f"{name}_visualize.log",
        )
        if visualize_status != 0:
            entry["status"] = "visualize_failed"
            entry["return_code"] = visualize_status
            write_state(state_path, state)
            if not args.continue_on_error:
                return visualize_status
            continue

        entry["status"] = "complete"
        write_state(state_path, state)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
