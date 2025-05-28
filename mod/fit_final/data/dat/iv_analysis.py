import pandas as pd
import matplotlib.pyplot as plt

def average_steady_state_iv(df, start_ms=290, end_ms=300):
    time = df["Time (ms)"]
    steady_df = df[(time >= start_ms) & (time <= end_ms)]
    mean_values = steady_df.drop(columns="Time (ms)").mean(axis=0)
    return pd.DataFrame({
        "Stimulus": mean_values.index,
        "SteadyState (nA or mV)": mean_values.values
    })

def peak_current_iv(df, search_start_ms=100, search_end_ms=120):
    time = df["Time (ms)"]
    peak_df = df[(time >= search_start_ms) & (time <= search_end_ms)]
    peak_values = peak_df.drop(columns="Time (ms)").max(axis=0)
    return pd.DataFrame({
        "Stimulus": peak_values.index,
        "Peak (nA or mV)": peak_values.values
    })

def plot_iv_curve(iv_df, steady_col="SteadyState (nA or mV)", peak_col="Peak (nA or mV)"):
    try:
        x = iv_df["Stimulus"].str.extract(r"([-+]?\d*\.?\d+)").astype(float).squeeze()
    except:
        x = pd.to_numeric(iv_df["Stimulus"], errors="coerce")

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

def validate_or_fix_time_vector(df, expected_interval_ms=0.02, verbose=True):
    """
    Checks the time vector for uniform sampling. Reconstructs it if irregular or missing.

    Parameters:
        df (pd.DataFrame): DataFrame with 'Time (ms)' and signal columns.
        expected_interval_ms (float): Expected sampling interval in milliseconds.
        verbose (bool): Print debug information.

    Returns:
        pd.DataFrame: DataFrame with valid 'Time (ms)'.
    """
    if "Time (ms)" not in df.columns:
        if verbose:
            print("⛔ 'Time (ms)' not found. Reconstructing time vector...")
        df.insert(0, "Time (ms)", np.arange(df.shape[0]) * expected_interval_ms)
        return df

    time = df["Time (ms)"]
    diffs = np.diff(time)

    if not np.allclose(diffs, expected_interval_ms, atol=1e-4):
        if verbose:
            print("⚠️ Irregular time vector detected. Reconstructing it.")
            print(f"Original mean interval: {np.mean(diffs):.6f} ms, std: {np.std(diffs):.6f}")
        df["Time (ms)"] = np.arange(len(time)) * expected_interval_ms
    else:
        if verbose:
            print(f"✅ Time vector OK: Δt = {np.mean(diffs):.4f} ms, std = {np.std(diffs):.6f}")

    return df
