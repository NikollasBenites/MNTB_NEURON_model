# data_loader.py

import matplotlib.pyplot as plt
from load_heka_python.load_heka import LoadHeka
import numpy as np


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
