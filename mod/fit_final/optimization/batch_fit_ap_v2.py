
import os
import glob
import subprocess
import re

# === Edit these paths for your environment
ap_dirs = [
    "/Users/nikollas/Library/CloudStorage/OneDrive-UniversityofSouthFlorida/MNTB_neuron/mod/fit_final/data/ap_P9_iMNTB",
    "/Users/nikollas/Library/CloudStorage/OneDrive-UniversityofSouthFlorida/MNTB_neuron/mod/fit_final/data/ap_P9_TeNT"
]

passive_root = "/Users/nikollas/Library/CloudStorage/OneDrive-UniversityofSouthFlorida/MNTB_neuron/mod/fit_final/figures"
output_root = "/Users/nikollas/Library/CloudStorage/OneDrive-UniversityofSouthFlorida/MNTB_neuron/mod/fit_final/results/_fit_ap"

fit_ap_script = "/Users/nikollas/Library/CloudStorage/OneDrive-UniversityofSouthFlorida/MNTB_neuron/mod/fit_final/optimization/fit_AP_v2.py"

def extract_cell_id(ap_filename):
    # Extracts date to cell ID portion (ignores current amplitude)
    match = re.search(r"\d{8}_P\d+_FVB_PunTeTx_.*?_S\d+C\d+", ap_filename)
    return match.group(0) if match else None

def find_passive_json(passive_root_dir, ap_filename):
    cell_id = extract_cell_id(ap_filename)
    if not cell_id:
        print(f"❌ Could not extract cell ID from: {ap_filename}")
        return None

    # Search all fit_passive_* folders
    candidate_dirs = glob.glob(os.path.join(passive_root_dir, f"fit_passive_*{cell_id}*"))
    for folder in candidate_dirs:
        json_files = glob.glob(os.path.join(folder, "passive_summary_*.json"))
        if json_files:
            return json_files[0]
    return None

def run_batch_fit():
    os.makedirs(output_root, exist_ok=True)
    for ap_dir in ap_dirs:
        for ap_path in glob.glob(os.path.join(ap_dir, "*.csv")):
            ap_file = os.path.basename(ap_path)
            print(f"🔍 Processing {ap_file}")
            passive_json = find_passive_json(passive_root, ap_file)
            if not passive_json:
                print(f"⚠️  No matching passive JSON found for: {ap_file}")
                continue

            output_dir = os.path.join(output_root, os.path.splitext(ap_file)[0])
            os.makedirs(output_dir, exist_ok=True)

            cmd = [
                "python", fit_ap_script,
                "--data", ap_path,
                "--passive_json", passive_json,
                "--output_dir", output_dir
            ]

            try:
                subprocess.run(cmd, check=True)
                print(f"✅ Finished: {ap_file}")
            except subprocess.CalledProcessError as e:
                print(f"❌ Error running fit_AP_v2 for {ap_file}: {e}")

if __name__ == "__main__":
    run_batch_fit()
