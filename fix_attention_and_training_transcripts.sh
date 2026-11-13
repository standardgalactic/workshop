#!/usr/bin/env bash
set -euo pipefail

python3 - "$@" <<'PY'
import argparse
import difflib
from pathlib import Path
import re
import shutil

parser = argparse.ArgumentParser(description="Correct the attention auction and local training transcripts.")
parser.add_argument("--dry-run", action="store_true", help="show changes without writing")
parser.add_argument("directory", nargs="?", type=Path, default=Path.cwd())
args = parser.parse_args()

corrections_by_stem = {
    "How_Algorithms_Auction_Your_Attention": {
        r"Platform control and the political economy of audience access": "Platform Control and the Political Economy of Audience Access",
        r"\bthe home tab\b": "the Home tab",
        r"\bThe tech starts by looking\b": "The text starts by looking",
        r"\ba second price auction\b": "a second-price auction",
        r"\btiny great sponsored label\b": "tiny gray sponsored label",
        r"\bbrute force those million impressions\b": "brute-forced those million impressions",
        r"\bthat spammers, all pages\b": "that spammer's alt pages",
    },
    "How_Local_Training_Warps_Global_AI": {
        r"\bcontrol variant\b": "control variate",
        r"\bControl variant\b": "Control variate",
        r"\balgorithms like scaffold\b": "algorithms like SCAFFOLD",
        r"^Scaffold\.(?=\s*$)": "SCAFFOLD.",
    },
}

def correct_line(line, rules):
    for pattern, replacement in rules:
        line = pattern.sub(replacement, line)
    return line

def correct_vtt(text, rules):
    blocks = re.split(r"(\r?\n(?:[ \t]*\r?\n)+)", text)
    for index in range(0, len(blocks), 2):
        lines = blocks[index].splitlines(keepends=True)
        for timing_index in (0, 1):
            if timing_index < len(lines) and re.match(r"^[\d:.]+ --> [\d:.]+", lines[timing_index]):
                blocks[index] = "".join(lines[:timing_index + 1] +
                                        [correct_line(line, rules) for line in lines[timing_index + 1:]])
                break
    return "".join(blocks)

def timings(text):
    return [line for line in text.splitlines(keepends=True) if " --> " in line]

changes = []
for stem, corrections in corrections_by_stem.items():
    rules = [(re.compile(pattern), replacement)
             for pattern, replacement in corrections.items()]
    for extension in ("txt", "vtt"):
        path = args.directory / f"{stem}.{extension}"
        original = path.read_bytes().decode("utf-8")
        corrected = (correct_vtt(original, rules) if extension == "vtt" else
                     "".join(correct_line(line, rules) for line in original.splitlines(keepends=True)))
        if extension == "vtt" and timings(original) != timings(corrected):
            raise SystemExit("WebVTT cue timings changed. No files were written.")
        changes.append((path, original, corrected))

for path, original, corrected in changes:
    if original == corrected:
        print(f"{path.name}: no changes")
    elif args.dry_run:
        for line in difflib.unified_diff(
            original.splitlines(), corrected.splitlines(),
            fromfile=path.name, tofile=f"{path.name} (corrected)", lineterm="",
        ):
            print(line)
    else:
        backup = path.with_name(path.name + ".bak")
        if not backup.exists():
            shutil.copy2(path, backup)
        path.write_bytes(corrected.encode("utf-8"))
        print(f"Corrected {path.name} (backup: {backup.name})")
PY
