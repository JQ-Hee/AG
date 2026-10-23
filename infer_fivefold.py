import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ag.data import load_fivefold_datasets
from ag.metrics import METRIC_NAMES, batch_metrics
from ag.runtime import load_model, resolve_device


def checkpoints_in(directory):
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    entries = manifest.get("checkpoints", [])
    expected = {f"model_{fold}.pt" for fold in range(5)}
    if (manifest.get("schema_version") != 1 or len(entries) != 5
            or {entry.get("path") for entry in entries} != expected):
        raise ValueError("Weights must list model_0.pt through model_4.pt in manifest.json")
    paths = [directory / f"model_{fold}.pt" for fold in range(5)]
    for path in paths:
        if not path.is_file():
            raise ValueError(f"Missing checkpoint file: {path.name}")
    return paths


def require_empty_output(path):
    if path.is_symlink() or (path.exists()
                            and (not path.is_dir() or any(path.iterdir()))):
        raise ValueError("Output must be a new or empty directory")


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def run_fold(dataset, checkpoint, output, device, num_workers):
    model = load_model(checkpoint, device)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers,
                        pin_memory=device.type == "cuda")
    records = []
    with torch.inference_mode():
        for raw in loader:
            batch = {key: value.to(device) if torch.is_tensor(value) else value
                     for key, value in raw.items()}
            prediction = model(batch["ct_t0"], batch["mask_t0"], batch["delta_t"])
            values = batch_metrics(prediction, batch)
            record = {
                "pair_id": raw["pair_id"][0],
                "patient_id": raw["patient_id"][0],
                "delta_t": float(batch["delta_t"].reshape(-1)[0].item()),
            }
            for name in METRIC_NAMES:
                value = float(values[name].reshape(-1)[0].item())
                if not np.isfinite(value):
                    raise ValueError(f"Non-finite {name} in fold {dataset.fold}")
                record[name] = value
            records.append(record)
    if len(records) != len(dataset) or not records:
        raise ValueError("The fold did not process every test pair")
    summary = {
        "fold": dataset.fold,
        "num_samples": len(records),
        "metrics": {name: float(np.mean([row[name] for row in records]))
                    for name in METRIC_NAMES},
    }
    output.mkdir()
    with (output / "per_pair.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    write_json(output / "metrics.json", summary)
    print(f"Fold {dataset.fold}: completed {len(records)} test pairs.", flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Private prepared dataset containing manifest.json, splits/, volumes/")
    parser.add_argument("--weights", type=Path, required=True,
                        help="Directory containing model_0.pt through model_4.pt and manifest.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args(argv)
    if args.num_workers < 0 or args.cpu_threads < 0:
        parser.error("Worker and CPU thread counts must be nonnegative")
    try:
        require_empty_output(args.output)
        device = resolve_device(args.device)
        datasets = load_fivefold_datasets(args.data_dir)
        checkpoints = checkpoints_in(args.weights)
    except (ValueError, RuntimeError, OSError) as error:
        parser.error(str(error))
    if args.cpu_threads:
        torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    require_empty_output(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    require_empty_output(args.output)
    folds = [
        run_fold(dataset, checkpoint, args.output / f"fold{fold}", device, args.num_workers)
        for fold, (dataset, checkpoint) in enumerate(zip(datasets, checkpoints))
    ]
    summary = {
        "num_folds": 5,
        "total_pairs": sum(fold["num_samples"] for fold in folds),
        "aggregation": "Unweighted fold mean and sample standard deviation (ddof=1)",
        "folds": folds,
        "metrics": {
            name: {"mean": float(np.mean([fold["metrics"][name] for fold in folds])),
                   "std": float(np.std([fold["metrics"][name] for fold in folds], ddof=1))}
            for name in METRIC_NAMES
        },
    }
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary["metrics"], indent=2), flush=True)


if __name__ == "__main__":
    main()
