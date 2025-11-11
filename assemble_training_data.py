"""
assemble_training_data.py
=========================
User-facing CLI to assemble a list of audio files across multiple emotional
speech datasets (EmoDB, RAVDESS, TESS). You can filter by gender and by
emotions (IDs, names, or dataset-specific aliases, depending on each parser’s
normalizer). The CLI delegates the heavy lifting to
`scripts.assembling_training_datasets.assembler`.

Typical layout expected (override with --root if needed):

    <repo>/
      assemble_training_data.py
      scripts/
        assembling_training_datasets/
          assembler.py
        training_datasets_parsing/
          EMODB.py
          RAVDESS.py
          TESS.py

The individual dataset parsers already implement robust normalization:
- EmoDB: accepts emotion IDs / codes / names.
- RAVDESS: accepts emotion IDs / names.
- TESS: accepts emotion IDs / names.

USAGE EXAMPLES
--------------

# 1) All datasets, only female speakers, emotions: happy or fear (by name)
$ python assemble_training_data.py --gender female --emotions happy fear

# 2) EmoDB + RAVDESS only, pass mixed emotion spec (id and name)
$ python assemble_training_data.py --datasets emodb ravdess --emotions 3 sad 6

# 3) Specify a custom dataset root (folder containing "dataset_training")
$ python assemble_training_data.py \
    --root /data/projects/Sentiment_PressConference/dataset_training \
    --emotions angry

# 4) Only TESS, any gender filter is ignored (TESS has only female speakers)
$ python assemble_training_data.py --datasets tess --emotions happy

Notes
-----
- Gender is respected by EmoDB and RAVDESS. For TESS it’s ignored with a
  warning (dataset is all female; older/younger adult female groups are
  handled within the TESS parser via "speaker" dimension).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Union, List

from scripts.assembling_training_datasets.assembler import (
    assemble_all_datasets,
    save_manifest,
    normalize_emotion_inputs,
    VALID_GENDERS,
)

###############################################################################
# Parser
###############################################################################

def _parse_args() -> argparse.Namespace:
    def emo_token(x: str):
        try:
            return int(x)
        except ValueError:
            return x.lower()
    parser = argparse.ArgumentParser(
        description="Assemble/select audio files from EMODB, RAVDESS, and TESS."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent / "dataset_training",
        help="Root dataset directory containing subfolders: EmoDB, RAVDESS, TESS "
             "(default: ./dataset_training)"
    )
    parser.add_argument(
        "--gender",
        type=str,
        choices=sorted(VALID_GENDERS),
        default=None,
        help="Optional gender filter: male or female"
    )
    parser.add_argument(
        "--emotions",
        nargs="+",
        type=emo_token,
        default=[1,2,3,4,5,6,7,8,9],
        help="Accepts either integer IDs or names."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Optional path to save a CSV manifest of selected files."
    )
    return parser.parse_args()

###############################################################################
# Main
###############################################################################

def main() -> None:
    args = _parse_args()
    # Normalize emotion inputs (mix of ints/strings → universal IDs)
    try:
        normalized_emotions: List[int] = normalize_emotion_inputs(args.emotions)
    except ValueError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(2)

    print(f"[INFO] Root: {args.root}")
    print(f"[INFO] Gender filter: {args.gender or '(none)'}")
    print(f"[INFO] Emotions (normalized universal IDs): {sorted(set(normalized_emotions))}")

    results = assemble_all_datasets(
        dataset_root=args.root,
        gender=args.gender,
        emotions=normalized_emotions,
    )

    total = sum(len(v) for v in results.values())
    print("\n[SUMMARY]")
    for k, v in results.items():
        print(f"  - {k:7s}: {len(v)} files")
    print(f"  - {'TOTAL':7s}: {total} files")

    if args.manifest:
        out_path = save_manifest(results, args.manifest)
        print(f"\n[INFO] Manifest written to: {out_path}")


if __name__ == "__main__":
    main()