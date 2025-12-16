"""
run_vomp_eval.py
====================

Description
-------------------------
Evaluate one or multiple emotion-prediction CSV files against the reference
Voice of Monetary Policy (VOMP) datasets, replicating the paper-style "voice tone"
metric and producing comparison outputs.

Structure
-------------------------------
- CLI interface with arguments:
    --vomp-dir     : directory containing VOMP_tones.csv and VOMP_classifications.csv
    --model-csvs   : one or more prediction CSV files (each with columns: item, emotion)
    --model-names  : optional names for each model (must match number of --model-csvs)
    --out-dir      : output directory (one subfolder per model)
    --plot         : if set, generate multi-model comparison plots

- Core steps:
    1. Validate VOMP reference files exist.
    2. Validate each model predictions CSV exists.
    3. Evaluate each model via `vomp_eval.eval_many_models(...)`:
       - load predictions
       - subset to VOMP timeframe
       - compute per-date voice tone and counts
       - merge with VOMP per-date data
       - compute aggregated metrics (All and by Chair if available)
    4. Save outputs per model:
       - tone_comp_grading.csv
       - tone_comp_metrics.csv
    5. Optionally produce multi-model plots in: <out-dir>/plots/

How to use with examples
------------------------

Single model:

    python run_vomp_eval.py \
        --vomp-dir   vomp_data/ \
        --model-csvs outputs/baseline_model/librosa/predictions_fomc.csv \
        --out-dir    outputs/vomp_eval \
        --plot

Multiple models:

    python run_vomp_eval.py \
        --vomp-dir    vomp_data/ \
        --model-csvs  outputs/baseline_model/librosa/predictions_fomc.csv \
        --model-names modelA modelB \
        --out-dir     outputs/vomp_eval \
        --plot
"""

###############
# Standard libraries
###############
import argparse
from pathlib import Path
import sys
from typing import List, Optional

###############
# Add project root to sys.path
###############
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

###############
# Project imports
###############
from scripts.visualization.tone_eval import (
    eval_many_models,
    save_results,
    plot_multi_models,
    VOMP_TONES_FILENAME,
    VOMP_CLASSIFICATIONS_FILENAME,
)

###############
# Argument parsing
###############
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate model prediction CSV(s) against VOMP reference metrics."
    )
    parser.add_argument(
        "--vomp-dir",
        type=Path,
        required=True,
        help="Directory containing VOMP_tones.csv and VOMP_classifications.csv.",
    )
    parser.add_argument(
        "--model-csvs",
        type=Path,
        nargs="+",
        required=True,
        help="One or more prediction CSV files (must include columns: item, emotion).",
    )
    parser.add_argument(
        "--model-names",
        type=str,
        nargs="*",
        default=None,
        help="Optional model names (must match number of --model-csvs).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs_vomp_eval"),
        help="Output directory. One subfolder per model will be created.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="If set, generate multi-model comparison plots.",
    )
    return parser.parse_args()


###############
# Validation helpers
###############
def _validate_inputs(
    vomp_dir: Path,
    model_csvs: List[Path],
    model_names: Optional[List[str]],
) -> None:
    """
    Validate required reference files and model inputs.

    - Checks VOMP reference files exist in `vomp_dir`.
    - Checks each predictions CSV exists.
    - If provided, checks model_names length matches model_csvs length.
    """
    tones_path = vomp_dir / VOMP_TONES_FILENAME
    class_path = vomp_dir / VOMP_CLASSIFICATIONS_FILENAME

    if not tones_path.exists():
        raise FileNotFoundError(f"Missing VOMP tones file: {tones_path}")
    if not class_path.exists():
        raise FileNotFoundError(f"Missing VOMP classifications file: {class_path}")

    for f in model_csvs:
        if not f.exists():
            raise FileNotFoundError(f"Missing predictions CSV: {f}")

    if model_names is not None:
        if len(model_names) == 0:
            return
        if len(model_names) != len(model_csvs):
            raise ValueError("If provided, --model-names must match number of --model-csvs.")


###############
# Main routine
###############
def main() -> None:
    args = _parse_args()

    ### Normalize optional model_names
    ## Treat empty list as None
    model_names: Optional[List[str]] = args.model_names
    if model_names is not None and len(model_names) == 0:
        model_names = None

    ### Validate inputs
    _validate_inputs(args.vomp_dir, args.model_csvs, model_names)

    ### Evaluate models
    results = eval_many_models(
        pred_csvs=args.model_csvs,
        vomp_dir=args.vomp_dir,
        model_names=model_names,
    )

    ### Save per-model outputs
    ## One folder per model under --out-dir
    for name, (merged_df, metrics_df, _) in results.items():
        model_out = args.out_dir / name
        save_results(merged_df, metrics_df, model_out)

        vt_mean = float(metrics_df.loc["Voice tone mean", "All_pred"]) if "All_pred" in metrics_df.columns else float("nan")
        print(f"[{name}] All_pred voice tone mean = {vt_mean:.6f}")

    if args.plot and len(results) >= 1:
        plot_multi_models(results, args.out_dir / "plots")


if __name__ == "__main__":
    main()
