#!/usr/bin/env python3
"""Rewrite a CrystFEL stream with sigma(I) divided by per-frame angle/score."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


REPORT_RE = re.compile(r"^unknown_image\s+//(\d+)\b.*\bangle/score=([^\s]+)")
EVENT_RE = re.compile(r"^Event:\s+//(\d+)\s*$")
REFLECTION_RE = re.compile(r"^(\s*[-+]?\d+\s+[-+]?\d+\s+[-+]?\d+\s+\S+\s+)(\S+)(.*)$")


def parse_report(path: Path) -> dict[str, float]:
    factors: dict[str, float] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            match = REPORT_RE.match(line)
            if not match:
                continue

            frame, value_text = match.groups()
            try:
                value = float(value_text)
            except ValueError as exc:
                raise ValueError(
                    f"{path}:{line_number}: cannot parse angle/score value {value_text!r}"
                ) from exc
            if value == 0.0:
                raise ValueError(f"{path}:{line_number}: angle/score is zero for frame //{frame}")
            if frame in factors:
                raise ValueError(f"{path}:{line_number}: duplicate report entry for frame //{frame}")
            factors[frame] = value

    if not factors:
        raise ValueError(f"No angle/score entries found in {path}")
    return factors


def format_float(value: float) -> str:
    return f"{value:.6f}"


def rewrite_stream(stream_path: Path, output_path: Path, factors: dict[str, float]) -> tuple[int, int]:
    events_seen: set[str] = set()
    current_frame: str | None = None
    current_factor: float | None = None
    in_reflections = False
    changed_sigmas = 0

    with stream_path.open("r", encoding="utf-8") as source, output_path.open(
        "w", encoding="utf-8"
    ) as target:
        for line_number, line in enumerate(source, start=1):
            event_match = EVENT_RE.match(line)
            if event_match:
                current_frame = event_match.group(1)
                try:
                    current_factor = factors[current_frame]
                except KeyError as exc:
                    raise KeyError(
                        f"{stream_path}:{line_number}: no angle/score value for Event: //{current_frame}"
                    ) from exc
                events_seen.add(current_frame)
                target.write(line)
                continue

            if "sigma(I)" in line and line.lstrip().startswith("h"):
                in_reflections = True
                target.write(line)
                continue

            if in_reflections and line.startswith("End of reflections"):
                in_reflections = False
                target.write(line)
                continue

            if in_reflections:
                if current_factor is None or current_frame is None:
                    raise RuntimeError(
                        f"{stream_path}:{line_number}: reflection table found before an Event line"
                    )

                reflection_match = REFLECTION_RE.match(line.rstrip("\n"))
                if reflection_match:
                    prefix, sigma_text, suffix = reflection_match.groups()
                    sigma = float(sigma_text)
                    new_sigma_text = format_float(sigma / current_factor)
                    target.write(prefix + new_sigma_text.rjust(len(sigma_text)) + suffix + "\n")
                    changed_sigmas += 1
                    continue

            target.write(line)

    missing_in_stream = set(factors).difference(events_seen)
    if missing_in_stream:
        examples = ", ".join(f"//{frame}" for frame in sorted(missing_in_stream)[:10])
        raise ValueError(
            f"{len(missing_in_stream)} report entries were not seen in the stream, e.g. {examples}"
        )

    return len(events_seen), changed_sigmas


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stream", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)

    factors = parse_report(args.report)
    event_count, changed_sigmas = rewrite_stream(args.stream, args.output, factors)

    print(f"Loaded {len(factors)} angle/score factors from {args.report}")
    print(f"Rewrote {event_count} stream events from {args.stream}")
    print(f"Divided {changed_sigmas} sigma(I) values")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
