# main.py

import subprocess
import os
import sys
import argparse
import pandas as pd
from datetime import datetime

def run_script(script_name, results_dir, age):
    print(f"\n🚀 Running: {script_name}")
    script_path = os.path.join(os.path.dirname(__file__), script_name)

    result = subprocess.run(
        [sys.executable, script_path, "--results_dir", results_dir, "--age", age],
        capture_output=True, text=True
    )

    if result.returncode != 0:
        print(f"❌ Error running {script_name}:\n{result.stderr}")
        sys.exit(1)
    else:
        print(f"✅ Finished: {script_name}")
        print(result.stdout)

def merge_all_results(results_dir):
    print(f"\n📊 Merging results from {results_dir}")
    summary = {}

    # Load Passive Params
    passive_path = os.path.join(results_dir, "best_fit_params.txt")
    with open(passive_path, "r") as f:
        gleak, gklt, gh, gka, erev, gkht, gna = map(float, f.read().strip().split(","))
        summary.update({
            "gleak": gleak, "gklt": gklt, "gh": gh, "gka": gka,
            "erev": erev, "gkht": gkht, "gna": gna
        })

    # Load Active Params
    active_path = os.path.join(results_dir, "all_fitted_params.csv")
    if os.path.exists(active_path):
        active_df = pd.read_csv(active_path)
        for col in active_df.columns:
            summary[col] = active_df.loc[0, col]

    # Load AP Features
    ap_feat_path = os.path.join(results_dir, "ap_features.csv")
    if os.path.exists(ap_feat_path):
        ap_df = pd.read_csv(ap_feat_path)
        for col in ap_df.columns:
            summary[f"AP_{col}"] = ap_df.loc[0, col]

    # Save Final Summary
    output_csv = os.path.join(os.path.dirname(__file__), "all_fit_summary.csv")
    if os.path.exists(output_csv):
        df = pd.read_csv(output_csv)
        df = pd.concat([df, pd.DataFrame([summary])], ignore_index=True)
    else:
        df = pd.DataFrame([summary])

    df.to_csv(output_csv, index=False)
    print(f"✅ Summary saved to: {output_csv}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--age", type=str, default="P9", help="Age label (default: P9)")
    args = parser.parse_args()

    age = args.age
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("results", f"BestFit_{age}_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    print(f"\n📁 Created results directory: {results_dir}")

    # Run pipeline
    run_script("fit_passive.py", results_dir, age)
    run_script("fit_AP.py", results_dir, age)
    run_script("fit_simulation.py", results_dir, age)

    # Merge
    merge_all_results(results_dir)

if __name__ == "__main__":
    main()
