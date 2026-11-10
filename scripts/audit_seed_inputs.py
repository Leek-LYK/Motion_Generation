"""Audit the downloaded SEED labels, metadata, splits, and extracted G1 files."""

import json
import math
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
METADATA = RAW / "bones_seed/metadata/seed_metadata_v004.parquet"
LABELS = RAW / "bones_seed/metadata/seed_metadata_v002_temporal_labels.jsonl"
SPLITS = RAW / "kimodo_benchmark/splits"
G1 = RAW / "bones_seed/g1"
REPORT = ROOT / "reports/seed_raw_audit.json"


def basename(value: str) -> str:
    """SEED split entries omit extensions and sometimes use Windows separators."""
    return Path(value.replace("\\", "/")).stem


def main() -> None:
    table = pq.read_table(METADATA)
    if "filename" not in table.column_names:
        raise RuntimeError("The metadata does not contain the expected filename column")
    metadata_names = Counter(str(x) for x in table["filename"].to_pylist() if x is not None)
    metadata_paths = set()
    if "move_g1_path" in table.column_names:
        metadata_paths = {
            str(value).replace("\\", "/")
            for value in table["move_g1_path"].to_pylist()
            if value is not None
        }

    labels = Counter()
    events = invalid_time = empty_text = 0
    with LABELS.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            labels[str(row["filename"])] += 1
            for event in row.get("events", []):
                events += 1
                try:
                    start = float(event["start_time"])
                    end = float(event["end_time"])
                    if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
                        invalid_time += 1
                except (KeyError, TypeError, ValueError):
                    invalid_time += 1
                if not str(event.get("description", "")).strip():
                    empty_text += 1

    split_sets = {}
    split_report = {}
    for path in sorted(SPLITS.glob("*.txt")):
        entries = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        split_sets[path.name] = set(entries)
        missing = [entry for entry in entries if basename(entry) not in labels]
        split_report[path.name] = {
            "entries": len(entries),
            "unique_paths": len(split_sets[path.name]),
            "missing_label_count": len(missing),
            "missing_label_paths": missing,
        }

    overlaps = {}
    filename_overlaps = {}
    for i, name in enumerate(sorted(split_sets)):
        for other in sorted(split_sets)[i + 1 :]:
            overlaps[f"{name} | {other}"] = len(split_sets[name] & split_sets[other])
            filename_overlaps[f"{name} | {other}"] = len(
                {basename(entry) for entry in split_sets[name]}
                & {basename(entry) for entry in split_sets[other]}
            )

    split_names = {basename(entry) for entries in split_sets.values() for entry in entries}
    expected_split_paths = {
        str(Path(path).relative_to("g1/csv").with_suffix(""))
        for path in metadata_paths
        if path.startswith("g1/csv/")
    }
    split_paths = set().union(*split_sets.values())

    report = {
        "metadata": {
            "rows": table.num_rows,
            "columns": table.column_names,
            "unique_filenames": len(metadata_names),
            "duplicate_filename_count": sum(n - 1 for n in metadata_names.values() if n > 1),
            "filenames_without_labels": len(set(metadata_names) - set(labels)),
            "label_filenames_without_metadata": len(set(labels) - set(metadata_names)),
        },
        "temporal_labels": {
            "rows": sum(labels.values()),
            "unique_filenames": len(labels),
            "duplicate_filename_count": sum(n - 1 for n in labels.values() if n > 1),
            "events": events,
            "invalid_time_events": invalid_time,
            "empty_description_events": empty_text,
        },
        "official_splits": split_report,
        "split_path_overlaps": overlaps,
        "split_filename_overlaps": filename_overlaps,
        "label_filenames_without_split": len(set(labels) - split_names),
        "split_filenames_without_label": len(split_names - set(labels)),
        "metadata_g1_paths_without_split": len(expected_split_paths - split_paths),
        "split_paths_without_metadata_g1": len(split_paths - expected_split_paths),
        "g1": {"extracted": G1.is_dir()},
    }

    if G1.is_dir():
        files = list(G1.rglob("*.csv"))
        g1_names = Counter(path.stem for path in files)
        file_paths = {str(path.relative_to(RAW / "bones_seed")) for path in files}
        report["g1"].update(
            csv_files=len(files),
            unique_filenames=len(g1_names),
            duplicate_filename_count=sum(n - 1 for n in g1_names.values() if n > 1),
            metadata_names_without_g1=len(set(metadata_names) - set(g1_names)),
            g1_names_without_metadata=len(set(g1_names) - set(metadata_names)),
            label_names_without_g1=len(set(labels) - set(g1_names)),
            metadata_paths_without_g1=len(metadata_paths - file_paths),
            g1_paths_without_metadata=len(file_paths - metadata_paths),
        )

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    temporary = REPORT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(REPORT)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
