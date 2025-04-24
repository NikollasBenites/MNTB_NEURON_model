import subprocess
import os
import sys
import pandas as pd


def run_script(script_name):
    script_path = os.path.join(os.path.dirname(__file__), script_name)
    print(f"\n🚀 Running: {script_name}")
    result = subprocess.run([sys.executable, script_path], capture_output=True, text=True)

    if result.returncode != 0:
        print(f"❌ Error running {script_name}:")
        print(result.stderr)
        sys.exit(1)
    else:
        print(f"✅ Finished: {script_name}")
        print(result.stdout)


def merge_all_results():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    passive_path = os.path.join(script_dir, "best_fit_params.txt")
    active_path = os.path.join(script_dir, "all_fitted_params.csv")
    sim_path = os.path.join(script_dir, "figures")  # base folder
    summary = {}

    # === Load Passive Params ===
    with open(passive_path, "r") as f:
        gleak, gklt, gh, erev = map(float, f.read().strip().split(","))
        summary.update({
            "gleak": gleak, "gklt": gklt, "gh": gh, "erev": erev
        })

    # === Load Active Params ===
    active_df = pd.read_csv(active_path)
    for col in active_df.columns:
        summary[col] = active_df.loc[0, col]

    # === Load AP Features from most recent folder ===
    sim_dirs = [f for f in os.listdir(sim_path) if f.startswith("BestFit")]
    if sim_dirs:
        latest_folder = max(sim_dirs)
        ap_feat_path = os.path.join(sim_path, latest_folder, "ap_features.csv")
        if os.path.exists(ap_feat_path):
            ap_df = pd.read_csv(ap_feat_path)
            for col in ap_df.columns:
                summary[f"AP_{col}"] = ap_df.loc[0, col]
        else:
            print("⚠️ AP features not found.")

    # === Save Final Summary ===
    output_csv = os.path.join(script_dir, "all_fit_summary.csv")
    if os.path.exists(output_csv):
        df = pd.read_csv(output_csv)
        df = pd.concat([df, pd.DataFrame([summary])], ignore_index=True)
    else:
        df = pd.DataFrame([summary])
    df.to_csv(output_csv, index=False)
    print(f"📊 Merged results saved to: {output_csv}")


def main():
    # Step 1: Passive Fit
    run_script("fit_passive.py")

    # Step 2: Active/AP Fit
    run_script("fit_AP.py")

    # Step 3: Simulation
    run_script("fit_simulation.py")

    # Step 4: Merge all results
    merge_all_results()


if __name__ == "__main__":
    main()
