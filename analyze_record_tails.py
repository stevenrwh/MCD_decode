#!/usr/bin/env python3
"""
Summarize the trailing 4-int tuples that appear inside each component record.

Usage:
    python analyze_record_tails.py --components FONTS/components_Mcalf020
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from font_components import iter_component_files


def collect_tail_stats(components_dir: Path) -> tuple[Counter[tuple[int, ...]], dict[tuple[int, ...], list[str]]]:
    counts: Counter[tuple[int, ...]] = Counter()
    samples: dict[tuple[int, ...], list[str]] = defaultdict(list)
    for component in iter_component_files(components_dir):
        for record in component.records:
            if len(record.values) < 4:
                continue
            tail = tuple(record.values[-4:])
            counts[tail] += 1
            if len(samples[tail]) < 8:
                samples[tail].append(component.label)
    return counts, samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect trailing record tuples inside component blobs.")
    parser.add_argument("--components", type=Path, required=True, help="Path to the directory with <index>_<label>.bin slices")
    parser.add_argument("--top", type=int, default=20, help="How many tuples to print (default 20)")
    return parser.parse_args()


def format_tuple(values: tuple[int, ...]) -> str:
    return "(" + ", ".join(f"{v:+d}" for v in values) + ")"


def main() -> int:
    args = parse_args()
    counts, samples = collect_tail_stats(args.components)
    print(f"Unique tail tuples: {len(counts)}")
    for tail, count in counts.most_common(args.top):
        names = ", ".join(samples[tail])
        print(f"{count:4d} x {format_tuple(tail)}  <- {names}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
