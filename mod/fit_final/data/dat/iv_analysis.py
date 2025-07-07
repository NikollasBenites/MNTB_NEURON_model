import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def average_steady_state_iv(df, start_ms=250, end_ms=300, first_stim=-100, sweep_step=10):
    time = df["Time (ms)"]
    steady_df = df[(time >= start_ms) & (time <= end_ms)]
    mean_values = steady_df.drop(columns="Time (ms)").mean(axis=0).values
    n_sweeps = len(mean_values)
    stimulus_vector = np.array([first_stim + i * sweep_step for i in range(n_sweeps)])
    return pd.DataFrame({
        "Stimulus": stimulus_vector,
        "SteadyState (nA or mV)": mean_values
    })

def peak_current_iv(df, search_start_ms=101, search_end_ms=150, first_stim=-100, sweep_step=10):
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


def latency_iv(df, search_start_ms=10, search_end_ms=300, dvdt_threshold=20, first_stim=-100, sweep_step=10):
    """
    Extract latency to first spike for each sweep in a dataframe.
    - search_start_ms: time after which to begin searching (to skip stimulus onset artifact)
    - dvdt_threshold: threshold for dV/dt to define spike onset
    """
    time = df["Time (ms)"].values
    signal_df = df.drop(columns="Time (ms)")
    latencies = []

    for col in signal_df.columns:
        v = signal_df[col].values
        t = time
        dvdt = np.gradient(v, t)

        try:
            start_idx = np.where(t >= search_start_ms)[0][0]
            end_idx = np.where(t >= search_end_ms)[0][0]
            dvdt_seg = dvdt[start_idx:end_idx]
            above_thresh = np.where(dvdt_seg > dvdt_threshold)[0]

            if len(above_thresh) == 0:
                latencies.append(np.nan)
            else:
                spike_idx = start_idx + above_thresh[0]
                latencies.append(t[spike_idx])
        except:
            latencies.append(np.nan)

    n_sweeps = len(latencies)
    stimulus_vector = np.array([first_stim + i * sweep_step for i in range(n_sweeps)])

    return pd.DataFrame({
        "Stimulus (pA)": stimulus_vector,
        "Latency (ms)": latencies
    })



def latency_iv_dual(df, search_start_ms=10.5, search_end_ms=200, dvdt_threshold=20,
                    first_stim=-100, sweep_step=10, peak_voltage_cutoff=-10):
    """
    Compute latency to spike threshold (dV/dt) and to AP peak for each sweep.
    Peak latency is detected using the first peak > peak_voltage_cutoff after threshold.

    Parameters:
        df: DataFrame with time and voltage sweeps (Time (ms) + N sweeps)
        search_start_ms: Start of latency search window (default: 10 ms)
        search_end_ms: End of latency search window (default: 200 ms)
        dvdt_threshold: Threshold in mV/ms for spike detection (default: 20)
        first_stim: First stimulus current in pA (default: -100)
        sweep_step: Current increment per sweep in pA (default: 10)
        peak_voltage_cutoff: Voltage required to define a real spike peak (default: -10 mV)

    Returns:
        DataFrame with columns:
        - Stimulus (pA)
        - Latency to Threshold (ms)
        - Latency to Peak (ms)
    """
    from scipy.signal import find_peaks
    time = df["Time (ms)"].values
    signal_df = df.drop(columns="Time (ms)")
    lat_thresh_list = []
    lat_peak_list = []

    for col in signal_df.columns:
        v = signal_df[col].values
        t = time
        dvdt = np.gradient(v, t)

        try:
            start_idx = np.where(t >= search_start_ms)[0][0]
            end_idx = np.where(t >= search_end_ms)[0][0]

            # === Threshold detection (dV/dt)
            dvdt_seg = dvdt[start_idx:end_idx]
            above_thresh = np.where(dvdt_seg > dvdt_threshold)[0]

            if len(above_thresh) > 0:
                spike_idx = start_idx + above_thresh[0]
                lat_thresh_list.append(t[spike_idx])

                # === Peak detection using first peak above cutoff
                v_seg = v[spike_idx:end_idx]
                t_seg = t[spike_idx:end_idx]

                peaks, properties = find_peaks(v_seg, height=peak_voltage_cutoff)
                if len(peaks) > 0:
                    peak_idx = peaks[0]  # First valid spike peak
                    lat_peak_list.append(t_seg[peak_idx])
                else:
                    lat_peak_list.append(np.nan)
            else:
                lat_thresh_list.append(np.nan)
                lat_peak_list.append(np.nan)

        except Exception:
            lat_thresh_list.append(np.nan)
            lat_peak_list.append(np.nan)

    n_sweeps = len(signal_df.columns)
    stimulus_vector = np.array([first_stim + i * sweep_step for i in range(n_sweeps)])

    return pd.DataFrame({
        "Stimulus (pA)": stimulus_vector,
        "Latency to Threshold (ms)": lat_thresh_list,
        "Latency to Peak (ms)": lat_peak_list
    })

