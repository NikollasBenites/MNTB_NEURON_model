
import matplotlib.pyplot as plt
from load_heka_python.load_heka import LoadHeka
import os
import numpy as np
import pandas as pd
from iv_analysis import average_steady_state_iv, peak_current_iv, plot_iv_curve, latency_iv, latency_iv_dual
from datetime import datetime
from contextlib import redirect_stdout
from scipy.optimize import curve_fit
import io
clip_duration_ms = 510
sampling_interval_ms = 0.02

def infer_mode_from_trace(voltage):
    """
    Infer recording mode by measuring the signal amplitude range.
    Assumes raw voltage signal is in Volts (V) or Amperes (A), depending on clamp mode.
    """
    if isinstance(voltage, list):
        all_values = np.concatenate(voltage)
    else:
        all_values = voltage

    max_val = np.max(all_values)
    min_val = np.min(all_values)
    range_val = max_val - min_val

    # Thresholds:
    # - ~0.005 V (5 mV) = likely current (voltage-clamp)
    # - ~0.020 V (20 mV) or more = likely voltage (current-clamp)
    if range_val > 0.01:  # >10 mV = voltage trace → current-clamp
        return "current_clamp"
    else:
        return "voltage_clamp"


def load_heka_data(file_path, group_idx, series_idx, channel_idx):
    with LoadHeka(file_path) as hf:
        hf.print_group_names()
        hf.print_series_names(group_idx=group_idx)
        series = hf.get_series_data(group_idx=group_idx, series_idx=series_idx, channel_idx=channel_idx,
                                    include_stim_protocol=True)

        voltage = series['data']
        time = series['time']
        stim = series.get('stim', None)
        labels = series.get('labels', None)
        n_sweeps = len(voltage)

        try:
            label_list = list(labels)
            if len(label_list) != n_sweeps:
                raise ValueError
        except:
            label_list = [None] * n_sweeps

    return voltage, time, stim, labels, n_sweeps, label_list, series

def extract_group_and_series_names(file_path):
    group_names = {}
    series_names = {}

    with LoadHeka(file_path) as hf:
        # Capture printed group names
        f = io.StringIO()
        with redirect_stdout(f):
            hf.print_group_names()
        group_lines = f.getvalue().strip().splitlines()

        for line in group_lines:
            # Example: "S1C1 (index: 0)"
            if "(index:" in line:
                name = line.split("(index:")[0].strip()
                index = int(line.split("(index:")[1].split(")")[0].strip())
                group_names[index] = name

        # Ask user to choose a group
        print("\n📁 Groups:")
        for idx, name in group_names.items():
            print(f"  [{idx}] {name}")
        group_idx = int(input("➡️ Enter group index: "))

        # Capture printed series names
        f = io.StringIO()
        with redirect_stdout(f):
            hf.print_series_names(group_idx=group_idx)
        series_lines = f.getvalue().strip().splitlines()

        for line in series_lines:
            # Example: "IV (index: 2)"
            if "(index:" in line:
                name = line.split("(index:")[0].strip()
                index = int(line.split("(index:")[1].split(")")[0].strip())
                series_names[index] = name

        # Ask user to choose a series
        print("\n📄 Series in group:")
        for idx, name in series_names.items():
            print(f"  [{idx}] {name}")
        series_idx = int(input("➡️ Enter series index: "))

    return group_idx, series_idx, group_names, series_names
def select_sweep(voltage, time, labels, is_vc):
    multiplier = 1e9 if is_vc else 1e3
    unit = "nA" if is_vc else "mV"
    n_sweeps = len(voltage)

    try:
        label_list = list(labels)
        if len(label_list) != n_sweeps:
            raise ValueError
    except:
        label_list = [None] * n_sweeps

    plt.figure(figsize=(12, 6))
    for i in range(n_sweeps):
        plt.plot(time[i] * 1000, voltage[i] * multiplier, label=f"Sweep {i}")

    plt.xlabel("Time (ms)")
    plt.ylabel(f"Signal ({unit})")
    plt.title("HEKA Sweeps - Inspect Before Fitting")
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print("\nAvailable sweeps:")
    for i in range(n_sweeps):
        stim_label = f"{label_list[i]} pA"
        print(f"  Sweep {i:2d} → {stim_label}")

    sweep_idx = int(sweep_rheobase-1)
    v_exp = voltage[sweep_idx] * multiplier
    t_exp = time[sweep_idx] * 1000

    print(f"\nSelected Sweep {sweep_idx}: Length = {len(v_exp)} samples")

    plt.figure()
    plt.plot(t_exp, v_exp, color='black')
    plt.xlabel("Time (ms)")
    plt.ylabel(f"Signal ({unit})")
    plt.title(f"Sweep {sweep_idx}")
    plt.tight_layout()
    plt.grid(True)

    return v_exp, t_exp, sweep_idx

# === Load data ===
# === TeNT ====
# 02062024_S4C1
# 12232024_S1C1
# 10142022_S1C1
# 12172022_S2C4
# 03232022_S1C2

# === iMNTB ====
# 12172022_S2C2
# 08122022_S2C1
# 08122022_S1C2
# 02072024_S3C3
# 08122022_S1C3

phenotype = "TeNT_S4C1"
sweep_step = 20
sweep_tau = 20 #pA
sweep_rheobase = 8

rheobase = int((sweep_rheobase - 5)*sweep_step)
rheobase_less1 = int((sweep_rheobase - 6)*sweep_step)
# group_idx = 3
# series_idx = 2
# channel_idx = 0
full_path_to_file = r"/Users/nikollas/Library/CloudStorage/OneDrive-UniversityofSouthFlorida/MNTB_neuron/mod/fit_final/data/dat/02062024_P9_FVB_PunTeTx.dat"
filename = os.path.splitext(os.path.basename(full_path_to_file))[0]
print(f"Loaded: {filename}")
group_idx, series_idx, group_names, series_names = extract_group_and_series_names(full_path_to_file)
voltage, time, stim, labels, n_sweeps, label_list, series = load_heka_data(
    full_path_to_file, group_idx, series_idx, channel_idx=0
)
print(f"\n✅ Loaded Group: {group_names[group_idx]} — Series: {series_names[series_idx]}")
mode = infer_mode_from_trace(voltage)
is_vc = (mode == "voltage_clamp")
print(f"📌 Detected mode from trace: {mode}")


multiplier = 1e9 if is_vc else 1e3
unit = "nA" if is_vc else "mV"
ylabel = f"Signal ({unit})"

plt.figure(figsize=(12, 6))
for i in range(n_sweeps):
    plt.plot(time[i] * 1000, voltage[i] * multiplier, label=f"Sweep {i}")

plt.xlabel("Time (ms)")
plt.ylabel(ylabel)
plt.title("HEKA Sweeps - Inspect Before Fitting")
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
plt.grid(True)
plt.tight_layout()
plt.show()

proceed = input("➡️ Do you want to select and save a specific sweep? (y/n): ").strip().lower()

if proceed == 'y':
    v_exp, t_exp, sweep_idx = select_sweep(voltage, time, labels, is_vc)

    output_dir = "../exported_sweeps"
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, f"full_sweep_{sweep_idx}_{filename}__data.csv")
    df = pd.DataFrame({
        "Time (ms)": t_exp,
        f"Signal ({unit})": v_exp
    })
    df.to_csv(output_file, index=False)
    print(f"\n✅ Sweep saved to: {output_file}")

    n_samples = int(clip_duration_ms / sampling_interval_ms)
    t_exp_clipped = t_exp[:n_samples]
    v_exp_clipped = v_exp[:n_samples]
    df_clipped = pd.DataFrame({
        "Time (ms)": t_exp_clipped,
        f"Signal ({unit})": v_exp_clipped
    })
    clipped_file = os.path.join(output_dir, f'sweep_{sweep_idx + 1}_clipped_{clip_duration_ms}ms_{filename}_{phenotype}_{rheobase-sweep_step}pA_{group_names[group_idx]}.csv')
    df_clipped.to_csv(clipped_file, index=False)
    print(f"✅ Clipped sweep saved to: {clipped_file}")

    plt.figure(figsize=(8, 4))
    plt.plot(t_exp_clipped, v_exp_clipped, color='darkblue')
    plt.xlabel("Time (ms)")
    plt.ylabel(ylabel)
    plt.title(f"Sweep {sweep_idx} — First {clip_duration_ms} ms")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
else:
    print("✅ Sweep selection skipped. Only overview plot generated.")

# === Save all sweeps
time_ms = time[0] * 1000
all_sweeps_df = pd.DataFrame({"Time (ms)": time_ms})
for i in range(n_sweeps):
    label = label_list[i]
    voltage_mv_or_na = voltage[i] * multiplier
    col_name = f"{label} pA" if label is not None else f"Sweep {i}"
    all_sweeps_df[col_name] = voltage_mv_or_na

output_dir = "../exported_sweeps"
os.makedirs(output_dir, exist_ok=True)
all_sweeps_file = os.path.join(output_dir, f"all_sweeps_{filename}_{phenotype}.csv")
all_sweeps_df.to_csv(all_sweeps_file, index=False)
print(f"\n✅ All sweeps saved to: {all_sweeps_file}")

# === Compute and plot IV curves
iv_steady = average_steady_state_iv(all_sweeps_df, sweep_step=sweep_step)
iv_peak = peak_current_iv(all_sweeps_df,sweep_step=sweep_step)
iv_combined = iv_steady.merge(iv_peak, on="Stimulus")
plot_iv_curve(iv_combined)

# === Save IV curve data ===
# Define output directory
iv_output_dir = "/Users/nikollas/Library/CloudStorage/OneDrive-UniversityofSouthFlorida/MNTB_neuron/mod/fit_final/data/fit_passive"
os.makedirs(iv_output_dir, exist_ok=True)

# Create a timestamp string: e.g., 20250605_1032
timestamp = datetime.now().strftime("%Y%m%d_%H%M")

# Generate filename with timestamp
iv_filename = f"experimental_data_{filename}_{phenotype}_{rheobase_less1}pA_{group_names[group_idx]}_{series_names[series_idx]}{series_idx}_{timestamp}.csv"
iv_output_path = os.path.join(iv_output_dir, iv_filename)

# Save the IV curve
iv_combined.to_csv(iv_output_path, index=False)
print(f"✅ IV curve data saved to: {iv_output_path}")

latency_df = latency_iv_dual(all_sweeps_df, search_start_ms=11, dvdt_threshold=35, first_stim=-100,
                             sweep_step=sweep_step)

# Plot
plt.figure(figsize=(6, 5))
plt.plot(latency_df["Stimulus (pA)"], latency_df["Latency to Threshold (ms)"], 'o-', label="To Threshold")
plt.plot(latency_df["Stimulus (pA)"], latency_df["Latency to Peak (ms)"], 's--', label="To Peak")
plt.xlabel("Injected Current (pA)")
plt.ylabel("Latency (ms)")
plt.title("Latency vs. Current Injection")
plt.grid(True)
plt.legend()
plt.tight_layout()

# === Save latency results to CSV ===
latency_output_dir = "/Users/nikollas/Library/CloudStorage/OneDrive-UniversityofSouthFlorida/MNTB_neuron/mod/fit_final/data/latency_results"
os.makedirs(latency_output_dir, exist_ok=True)

latency_filename = f"latency_data_{filename}_{phenotype}.csv"
latency_output_path = os.path.join(latency_output_dir, latency_filename)

latency_df.to_csv(latency_output_path, index=False)
print(f"✅ Latency data saved to: {latency_output_path}")
# === Save the figure as PNG and PDF ===
fig_filename_base = f"latency_plot_{filename}_{phenotype}"
fig_path_png = os.path.join(latency_output_dir, fig_filename_base + ".png")
fig_path_pdf = os.path.join(latency_output_dir, fig_filename_base + ".pdf")

plt.savefig(fig_path_png, dpi=300)
plt.savefig(fig_path_pdf)
print(f"✅ Plot saved as: {fig_path_png} and .pdf")
plt.show()
#
# if not is_vc:
#     print("\n🔍 Estimating τm and cm from clipped voltage trace...")
#
#     # === Fit exponential to voltage decay
#     def exp_decay(t, V0, tau, Vinf):
#         return Vinf + (V0 - Vinf) * np.exp(-t / tau)
#
#     # Use a short window after stimulus onset (assume stimulus starts at ~50 ms)
#     fit_window_start = 12  # ms
#     fit_window_end = 100  # ms
#
#     idx_start = np.searchsorted(t_exp_clipped, fit_window_start)
#     idx_end = np.searchsorted(t_exp_clipped, fit_window_end)
#
#     t_fit = t_exp_clipped[idx_start:idx_end]
#     v_fit = v_exp_clipped[idx_start:idx_end]
#
#     try:
#         popt, _ = curve_fit(exp_decay, t_fit, v_fit, p0=[v_fit[0], 10, v_fit[-1]])
#         V0, tau_m, Vinf = popt
#
#         print(f"⏱️  Fitted τm = {tau_m:.2f} ms")
#
#         # === Estimate input resistance Rm
#         I_step = sweep_step * 1e-12  # in Amperes
#         delta_V = (Vinf - V0) * 1e-3  # Convert mV to Volts
#         Rm = abs(delta_V / I_step)  # Ohms
#
#         print(f"🔌 Estimated Rm = {Rm * 1e-6:.2f} MΩ")
#
#         # === Compute Cm
#         Cm = tau_m / (Rm * 1e3)  # Farads
#         soma_area = 8.427e-6  # cm² (25 pF if cm = 1 uF/cm²)
#         cm = Cm / soma_area * 1e6  # convert to μF/cm²
#
#         print(f"🧪 Estimated Cm = {Cm * 1e12:.2f} pF")
#         print(f"📐 Estimated cm = {cm:.2f} μF/cm²")
#
#     except Exception as e:
#         print(f"⚠️ Could not fit exponential to estimate τm/cm: {e}")
