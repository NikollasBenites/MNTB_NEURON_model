
import matplotlib.pyplot as plt
from load_heka_python.load_heka import LoadHeka
import os
import numpy as np
import pandas as pd
from iv_analysis import average_steady_state_iv, peak_current_iv, plot_iv_curve

clip_duration_ms = 510
sampling_interval_ms = 0.02

def is_voltage_clamp(trace_array):
    return np.max(np.abs(trace_array)) < 1.0

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
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print("\nAvailable sweeps:")
    for i in range(n_sweeps):
        stim_label = f"{label_list[i]} pA"
        print(f"  Sweep {i:2d} → {stim_label}")

    sweep_idx = int(input(f"\nSelect sweep index (0 to {n_sweeps - 1}): "))
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
full_path_to_file = r"/Users/nikollas/Library/CloudStorage/OneDrive-UniversityofSouthFlorida/MNTB_neuron/mod/fit_final/data/dat/02012023_P4_FVB_PunTeTx.dat"
filename = os.path.splitext(os.path.basename(full_path_to_file))[0]

voltage, time, stim, labels, n_sweeps, label_list, series = load_heka_data(
    full_path_to_file, group_idx=1, series_idx=2, channel_idx=0
)

is_vc = is_voltage_clamp(voltage[1])
multiplier = 1e9 if is_vc else 1e3
unit = "nA" if is_vc else "mV"
ylabel = f"Signal ({unit})"

plt.figure(figsize=(12, 6))
for i in range(n_sweeps):
    plt.plot(time[i] * 1000, voltage[i] * multiplier, label=f"Sweep {i}")

plt.xlabel("Time (ms)")
plt.ylabel(ylabel)
plt.title("HEKA Sweeps - Inspect Before Fitting")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
plt.grid(True)
plt.tight_layout()
plt.show()

proceed = input("➡️ Do you want to select and save a specific sweep? (y/n): ").strip().lower()

if proceed == 'y':
    v_exp, t_exp, sweep_idx = select_sweep(voltage, time, labels, is_vc)

    output_dir = "../exported_sweeps"
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, f"full_sweep_{sweep_idx}_{filename}_data.csv")
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
    clipped_file = os.path.join(output_dir, f"sweep_{sweep_idx}_clipped_{clip_duration_ms}ms_{filename}.csv")
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
all_sweeps_file = os.path.join(output_dir, f"all_sweeps_{filename}_tonic_TeNTx.csv")
all_sweeps_df.to_csv(all_sweeps_file, index=False)
print(f"\n✅ All sweeps saved to: {all_sweeps_file}")

# === Compute and plot IV curves
iv_steady = average_steady_state_iv(all_sweeps_df)
iv_peak = peak_current_iv(all_sweeps_df)
iv_combined = iv_steady.merge(iv_peak, on="Stimulus")
plot_iv_curve(iv_combined)
