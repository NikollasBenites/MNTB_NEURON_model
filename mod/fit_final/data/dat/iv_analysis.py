import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def average_steady_state_iv(df, start_ms=290, end_ms=300, first_stim=-100, sweep_step=10):
    time = df["Time (ms)"]
    steady_df = df[(time >= start_ms) & (time <= end_ms)]
    mean_values = steady_df.drop(columns="Time (ms)").mean(axis=0).values
    n_sweeps = len(mean_values)
    stimulus_vector = np.array([first_stim + i * sweep_step for i in range(n_sweeps)])
    return pd.DataFrame({
        "Stimulus": stimulus_vector,
        "SteadyState (nA or mV)": mean_values
    })

def peak_current_iv(df, search_start_ms=20, search_end_ms=250, first_stim=-100, sweep_step=10):
    time = df["Time (ms)"]
    peak_df = df[(time >= search_start_ms) & (time <= search_end_ms)]
    peak_values = peak_df.drop(columns="Time (ms)").max(axis=0).values
    n_sweeps = len(peak_values)
    stimulus_vector = np.array([first_stim + i * sweep_step for i in range(n_sweeps)])
    return pd.DataFrame({
        "Stimulus": stimulus_vector,
        "Peak (nA or mV)": peak_values
    })

def plot_iv_curve(iv_df, steady_col="SteadyState (nA or mV)", peak_col="Peak (nA or mV)"):
    x = iv_df["Stimulus"]
    y_steady = iv_df[steady_col]
    y_peak = iv_df[peak_col]

    plt.figure(figsize=(6, 4))
    plt.plot(x, y_steady, marker='o', label="Steady-State")
    plt.plot(x, y_peak, marker='x', label="Peak")
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    plt.xlabel("Command Step (pA or mV)")
    plt.ylabel("Response (nA or mV)")
    plt.title("IV Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
