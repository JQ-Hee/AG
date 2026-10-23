import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .io import load_volume, normalize_ct


def _unique_object(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("Duplicate field in a dataset JSON file")
        result[key] = value
    return result


def _relative_file(root, relative):
    if not isinstance(relative, str) or not relative:
        raise ValueError("Dataset file paths must be nonempty relative strings")
    path = Path(relative)
    if path.is_absolute():
        raise ValueError("Dataset file paths must be relative")
    resolved = (root / path).resolve()
    if root not in resolved.parents:
        raise ValueError("Dataset file paths must remain inside the data directory")
    if not resolved.is_file():
        raise ValueError("A required dataset file is missing or is not a regular file")
    return resolved


def _read_json(root, relative):
    with _relative_file(root, relative).open(encoding="utf-8") as handle:
        result = json.load(handle, object_pairs_hook=_unique_object)
    if not isinstance(result, dict):
        raise ValueError("Dataset JSON files must contain objects")
    return result


def _identifier(value):
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError("Patient and sequence identifiers must be nonempty strings without surrounding whitespace")
    return value


class TestPairsDataset(Dataset):

    def __init__(self, root, fold, pair_records):
        self.root = root
        self.fold = fold
        self.pair_records = pair_records

    def __len__(self):
        return len(self.pair_records)

    def __getitem__(self, index):
        pair = self.pair_records[index]
        ct0, mask0 = load_volume(_relative_file(self.root, pair["baseline"]))
        ct1, mask1 = load_volume(_relative_file(self.root, pair["followup"]))
        arrays = (normalize_ct(ct0), normalize_ct(ct1), mask0, mask1)
        tensors = [torch.from_numpy(np.ascontiguousarray(array[None], dtype=np.float32))
                   for array in arrays]
        return {
            "ct_t0": tensors[0], "ct_t1": tensors[1],
            "mask_t0": tensors[2], "mask_t1": tensors[3],
            "delta_t": torch.tensor([pair["interval"]], dtype=torch.float32),
            "pair_id": pair["pair_id"], "patient_id": pair["patient_id"],
        }


def load_fivefold_datasets(data_dir):
    root = Path(data_dir).expanduser().resolve()
    if not root.is_dir():
        raise ValueError("The data directory does not exist")
    manifest = _read_json(root, "manifest.json")
    if type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1:
        raise ValueError("Expected dataset schema_version=1")
    if manifest.get("axis_order") != "zyx" or manifest.get("spacing_mm") != [1.0, 1.0, 1.0]:
        raise ValueError("Expected zyx arrays with 1 mm isotropic spacing")
    if manifest.get("ct_units") != "HU" or manifest.get("pair_protocol") != "adjacent":
        raise ValueError("Expected HU CT inputs and adjacent observation pairs")
    sequences = manifest.get("sequences")
    if not isinstance(sequences, list) or not sequences:
        raise ValueError("The manifest must contain a nonempty sequence list")
    patients, sequence_ids, volume_paths = set(), set(), set()
    records = []
    for sequence in sequences:
        if not isinstance(sequence, dict):
            raise ValueError("Each sequence must be an object")
        patient = _identifier(sequence.get("patient_id"))
        sequence_id = _identifier(sequence.get("sequence_id"))
        if sequence_id in sequence_ids:
            raise ValueError("Duplicate sequence identifier in the manifest")
        patients.add(patient)
        sequence_ids.add(sequence_id)
        visits = sequence.get("timepoints")
        intervals = sequence.get("intervals_months")
        if (not isinstance(visits, list) or len(visits) < 2
                or not isinstance(intervals, list) or len(intervals) != len(visits) - 1):
            raise ValueError("Each sequence needs at least two observations and one interval per adjacent pair")
        paths = []
        for visit in visits:
            if not isinstance(visit, dict):
                raise ValueError("Each observation must be an object")
            path = visit.get("path")
            resolved = _relative_file(root, path)
            if resolved in volume_paths:
                raise ValueError("A volume file is indexed more than once")
            volume_paths.add(resolved)
            paths.append(path)
        for index, interval in enumerate(intervals):
            if (type(interval) not in (int, float) or interval < 1
                    or interval > float(np.finfo(np.float32).max) or not math.isfinite(interval)):
                raise ValueError("Intervals must be finite float32-representable numbers of at least one month")
            records.append({
                "patient_id": patient, "sequence_id": sequence_id,
                "pair_index": index, "pair_id": "%s-P%03d" % (sequence_id, index),
                "baseline": paths[index], "followup": paths[index + 1],
                "interval": float(interval),
            })
    memberships, seen = [], set()
    for fold in range(5):
        payload = _read_json(root, "splits/fold%d.json" % fold)
        if (type(payload.get("schema_version")) is not int or payload["schema_version"] != 1
                or type(payload.get("fold")) is not int or payload["fold"] != fold
                or payload.get("split_by") != "patient_id"):
            raise ValueError("Invalid patient-fold metadata")
        splits = payload.get("splits")
        members = splits.get("test") if isinstance(splits, dict) else None
        if not isinstance(members, list) or not members:
            raise ValueError("Each fold must contain a nonempty test patient list")
        members = [_identifier(value) for value in members]
        membership = set(members)
        if len(membership) != len(members) or seen & membership:
            raise ValueError("Duplicate test patient within or across folds")
        if not membership.issubset(patients):
            raise ValueError("A test fold contains patients absent from the manifest")
        seen.update(membership)
        memberships.append(membership)
    if seen != patients:
        raise ValueError("The five test folds must cover every manifest patient exactly once")
    return [TestPairsDataset(root, fold, [record for record in records
                                        if record["patient_id"] in membership])
            for fold, membership in enumerate(memberships)]
