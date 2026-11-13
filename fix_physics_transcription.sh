#!/usr/bin/env bash
set -euo pipefail

python3 - "$@" <<'PY'
import argparse
import difflib
from pathlib import Path
import re
import shutil

parser = argparse.ArgumentParser(description="Correct the Physics Without a Spacetime Grid transcripts.")
parser.add_argument("--dry-run", action="store_true", help="show changes without writing")
parser.add_argument("directory", nargs="?", type=Path, default=Path.cwd())
args = parser.parse_args()

corrections = {
    r"\bLucretia Rivera\b": "Lucrezia Ravera",
    r"\bRivera\b": "Ravera",
    r"\bRavelli(?= says, take away)":"Ravera",
    r"\bRavelli\b": "Rovelli",
    r"\bFrancois\b": "François",
    r"\b(?:Kurt J\. Mungel|Haimengal|Jay Mangal)\b": "Curt Jaimungal",
    r"\bFleikian\b": "Flyxion",
    r"\bCouchy\b": "Cauchy",
    r"\bwhole argument\b": "hole argument",
    r"\ba whole(?=\.)": "a hole",
    r"\b(inside|outside|boundary of|contents of) the whole\b": r"\1 the hole",
    r"\bworld universe\b": "swirled universe",
    r"\bbundled geometry\b": "bundle geometry",
    r"\b(?:Griebov|Grubov)\b": "Gribov",
    r"\bFadiyev (?:Popov|pop-off)\b": "Faddeev–Popov",
    r"\baddressed variables\b": "dressed variables",
    r"\bYou're dressing as a perfect, adequate interface\b": "Your dressing is a perfect, adequate interface",
    r"\bdress (theory|configuration space)\b": r"dressed \1",
    r"\bdelta quant\b": "delta cont",
    r"\bbackground referencers\b": "background references",
    r"\bmaintain distinction means\b": "maintained distinction means",
}
rules = [(re.compile(pattern, re.MULTILINE), replacement)
         for pattern, replacement in corrections.items()]
stem = "Physics_Without_a_Spacetime_Grid"

def correct_line(line):
    for pattern, replacement in rules:
        line = pattern.sub(replacement, line)
    return line

def correct_vtt(text):
    blocks = re.split(r"(\r?\n(?:[ \t]*\r?\n)+)", text)
    for index in range(0, len(blocks), 2):
        lines = blocks[index].splitlines(keepends=True)
        for timing_index in (0, 1):
            if timing_index < len(lines) and re.match(r"^[\d:.]+ --> [\d:.]+", lines[timing_index]):
                blocks[index] = "".join(lines[:timing_index + 1] +
                                        [correct_line(line) for line in lines[timing_index + 1:]])
                break
    return "".join(blocks)

def timings(text):
    return [line for line in text.splitlines(keepends=True) if " --> " in line]

changes = []
for extension in ("txt", "vtt"):
    path = args.directory / f"{stem}.{extension}"
    original = path.read_bytes().decode("utf-8")
    corrected = (correct_vtt(original) if extension == "vtt" else
                 "".join(correct_line(line) for line in original.splitlines(keepends=True)))
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
