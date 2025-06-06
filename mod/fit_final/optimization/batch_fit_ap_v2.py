
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

fit_ap_script = "/Users/nikollas/Library/CloudStorage/OneDrive-UniversityofSouthFlorida/MNTB_neuron/mod/fit_final/optimization/fit_AP_v2_updated.py"

def extract_cell_parts(ap_filename):
    """
    From: sweep_14_clipped_510ms_08122022_P9_FVB_PunTeTx_iMNTB_180pA_S2C1.csv
    Extract:
        - part1 = 08122022_P9_FVB_PunTeTx
        - part2 = iMNTB or TeNT
        - part3 = S2C1
    """
    base = os.path.basename(ap_filename).replace(".csv", "")
    parts = base.split("_")

    try:
        date_idx = next(i for i, p in enumerate(parts) if re.match(r"^\d{8}$", p))
        stim_idx = next(i for i, p in enumerate(parts) if re.match(r"^\d+pA$", p))
        sc_idx = stim_idx + 1 if (stim_idx + 1) < len(parts) else None

        part1 = "_".join(parts[date_idx:date_idx+4])   # date + P9 + FVB + PunTeTx
        part2 = parts[stim_idx - 1]                    # group (iMNTB or TeNT)
        part3 = parts[sc_idx] if sc_idx else None      # S2C1

        return part1, part2, part3
    except Exception as e:
        print(f"❌ Failed to extract parts: {e}")
        return None, None, None


def find_passive_json(passive_root, ap_filename):
    part1, part2, part3 = extract_cell_parts(ap_filename)
    if not all([part1, part2, part3]):
        print(f"❌ Incomplete parts for: {ap_filename}")
        return None

    print(f"🔍 Matching passive JSON for: part1 = {part1}, part2 = {part2}, part3 = {part3}")

    for root, dirs, files in os.walk(passive_root):
        for file in files:
            if file.endswith(".json") and all(p in file for p in [part1, part2, part3]):
                print(f"✅ Match found: {file}")
                return os.path.join(root, file)

    print(f"⚠️ No passive JSON match for: {ap_filename}")
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
                "--ap_file", ap_path,
                "--passive_json", passive_json,
                "--output_root", output_dir
            ]
            print("📤 Running command:", " ".join(cmd))
            try:
                subprocess.run(cmd, check=True)
                print(f"✅ Finished: {ap_file}")
            except subprocess.CalledProcessError as e:
                print(f"❌ Error running fit_AP_v2 for {ap_file}: {e}")

if __name__ == "__main__":
    run_batch_fit()
