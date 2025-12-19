#!/usr/bin/env python3
"""
Extract per-glyph header fields from the sliced CComponentDefinition blobs.

For each `FONTS/components/*.bin` file we record:
  * The glyph label (e.g., M92A)
  * Header ints 16-27 (converted to floats by dividing by 32768)
  * The record count implied by the header vs. actual body records
  * The bounding box of the normalized coordinates stored in the records

This CSV gives us raw material to reverse width/height/baseline formulas without
having to keep spelunking in hex dumps.
"""

from __future__ import annotations

import argparse
import csv
import struct
from pathlib import Path

from font_components import ComponentDefinition, iter_component_files


def write_csv(rows: list[dict], destination: Path) -> None:
    fieldnames = [
        "label",
        "record_count_header",
        "record_count_body",
        "bbox_min_x",
        "bbox_min_y",
        "bbox_max_x",
        "bbox_max_y",
    ]
    # Append header[16:27] as float columns.
    header_cols = [f"h{i}" for i in range(16, 27)]
    fieldnames.extend(header_cols)

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            bbox = row["bbox"]
            header_slice = row["header"][16:27]
            floats = [v / 32768.0 for v in header_slice]
            payload = {
                "label": row["label"],
                "record_count_header": row["record_count_header"],
                "record_count_body": row["record_count_body"],
                "bbox_min_x": bbox[0],
                "bbox_min_y": bbox[1],
                "bbox_max_x": bbox[2],
                "bbox_max_y": bbox[3],
            }
            for idx, col in enumerate(header_cols):
                payload[col] = floats[idx] if idx < len(floats) else None
            writer.writerow(payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize MAIN font component headers.")
    parser.add_argument(
        "--components-dir",
        type=Path,
        default=Path("FONTS/components"),
        help="Directory containing <index>_<label>.bin slices (default: FONTS/components)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("FONTS/component_header_stats.csv"),
        help="Destination CSV path (default: FONTS/component_header_stats.csv)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = []
    for definition in iter_component_files(args.components_dir):
        bbox = definition.bounding_box()
        bbox_vals = bbox if bbox else (None, None, None, None)
        rows.append(
            {
                "label": definition.label,
                "header": definition.header,
                "record_count_header": definition.record_count_header,
                "record_count_body": len(definition.records),
                "bbox": bbox_vals,
            }
        )
    write_csv(rows, args.output)
    print(f"Wrote {len(rows)} component summaries to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
