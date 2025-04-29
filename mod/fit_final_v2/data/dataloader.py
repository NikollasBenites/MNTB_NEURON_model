# data_loader.py

import matplotlib.pyplot as plt
from load_heka_python.load_heka import LoadHeka
import numpy as np
clip = 1
saveclip = 1

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

    return voltage, time, stim, labels


def select_sweep(voltage, time, labels):
    n_sweeps = len(voltage)

    # Validate labels
    try:
        label_list = list(labels)
        if len(label_list) != n_sweeps:
            raise ValueError
    except:
        label_list = [None] * n_sweeps

    # Plot all sweeps (optional for quick inspection)
    plt.figure(figsize=(12, 6))
    for i in range(n_sweeps):
        stim_label = f"{label_list[i]} pA" if label_list[i] is not None else f"Sweep {i}"
        plt.plot(time[i] * 1000, voltage[i], label=f"Sweep {i}")

    plt.xlabel("Time (ms)")
    plt.ylabel("Membrane potential (mV)")
    plt.title("HEKA Sweeps - Inspect Before Fitting")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.grid(True)
    plt.tight_layout()

    # Print sweep options
    print("\nAvailable sweeps:")
    for i in range(n_sweeps):
        stim_label = f"{label_list[i]} pA" if label_list[i] is not None else "unknown"
        print(f"  Sweep {i:2d} → {stim_label}")

    sweep_idx = int(input(f"\nSelect sweep index (0 to {n_sweeps - 1}): "))
    v_exp = voltage[sweep_idx]*1000
    t_exp = time[sweep_idx] * 1000  # ms

    print(f"\nSelected Sweep {sweep_idx}: Length = {len(v_exp)} samples")

    # Plot selected sweep
    plt.figure()
    plt.plot(t_exp, v_exp, color='black')
    plt.xlabel("Time (ms)")
    plt.ylabel("Voltage (mV)")
    plt.title(f"Sweep {sweep_idx}")
    plt.tight_layout()
    plt.grid(True)

    return v_exp, t_exp, sweep_idx

full_path_to_file = r"/mod/fit_final/data/dat/12232024_P9_FVB_PunTeTx_Dan.dat"
voltage, time, stim, labels = load_heka_data(full_path_to_file, group_idx=0, series_idx=2, channel_idx=0)
v_exp, t_exp, sweep_idx = select_sweep(voltage, time, labels)

import pandas as pd
import os

# Create a folder to save if it doesn't exist
output_dir = "exported_sweeps"
os.makedirs(output_dir, exist_ok=True)

# Define file name
output_file = os.path.join(output_dir, f"sweep_{sweep_idx}_data.csv")

# Combine time and voltage into a DataFrame
df = pd.DataFrame({
    "Time (ms)": t_exp,
    "Voltage (mV)": v_exp
})

# Save to CSV
df.to_csv(output_file, index=False)
print(f"\n✅ Sweep saved to: {output_file}")

if clip == 1:
    # Clip the first x ms
    clip_duration_ms = 510
    sampling_interval_ms = 0.02
    n_samples = int(clip_duration_ms / sampling_interval_ms)

    t_exp_clipped = t_exp[:n_samples]
    v_exp_clipped = v_exp[:n_samples]

    # Combine into DataFrame
    df_clipped = pd.DataFrame({
        "Time (ms)": t_exp_clipped,
        "Voltage (mV)": v_exp_clipped
    })
if saveclip == 1:
    # Save clipped trace
    clipped_file = os.path.join(output_dir, f"sweep_{sweep_idx}_clipped_{clip_duration_ms}ms.csv")
    df_clipped.to_csv(clipped_file, index=False)
    print(f"✅ Clipped sweep (first {clip_duration_ms} ms) saved to: {clipped_file}")

# Plot the clipped sweep
plt.figure(figsize=(8, 4))
plt.plot(t_exp_clipped, v_exp_clipped, color='darkblue')
plt.xlabel("Time (ms)")
plt.ylabel("Voltage (mV)")
plt.title(f"Sweep {sweep_idx} — First {clip_duration_ms} ms")
plt.grid(True)
plt.tight_layout()
plt.show()
