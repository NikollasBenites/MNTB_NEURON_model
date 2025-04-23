import subprocess
import os
import sys


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


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    passive_params_path = os.path.join(base_dir, "best_fit_params.txt")
    full_params_path = os.path.join(base_dir, "all_fitted_params.csv")

    # Step 1: Run Passive Fit
    run_script("fit_passive.py")
    if not os.path.exists(passive_params_path):
        print("❌ Passive fit output not found!")
        sys.exit(1)

    # Step 2: Run AP Fit
    run_script("fit_AP.py")
    if not os.path.exists(full_params_path):
        print("❌ All fitted params not found!")
        sys.exit(1)

    # Step 3: Run Simulation
    run_script("fit_simulation.py")


if __name__ == "__main__":
    main()
