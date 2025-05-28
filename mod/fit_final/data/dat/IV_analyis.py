import pandas as pd
import matplotlib.pyplot as plt

def average_steady_state_iv(df, start_ms=290, end_ms=300):
    """
    Compute the steady-state IV curve by averaging signal from a defined window.

    Parameters:
        df (pd.DataFrame): Input DataFrame with 'Time (ms)' and signal columns.
        start_ms (float): Start of steady-state window.
        end_ms (float): End of steady-state window.

    Returns:
        pd.DataFrame: IV curve as mean value per stimulus step in steady state.
    """
    time = df["Time (ms)"]
    steady_df = df[(time >= start_ms) & (time <= end_ms)]
    mean_values = steady_df.drop(columns="Time (ms)").mean(axis=0)
    return pd.DataFrame({
        "Stimulus": mean_values.index,
        "SteadyState (nA or mV)": mean_values.values
    })


def peak_current_iv(df, search_start_ms=100, search_end_ms=120):
    """
    Compute the IV curve from the peak (min) response in a defined time window.

    Parameters:
        df (pd.DataFrame): Input DataFrame with 'Time (ms)' and signal columns.
        search_start_ms (float): Start of peak search window.
        search_end_ms (float): End of peak search window.

    Returns:
        pd.DataFrame: IV curve with peak (min) value per stimulus step.
    """
    time = df["Time (ms)"]
    peak_df = df[(time >= search_start_ms) & (time <= search_end_ms)]
    peak_values = peak_df.drop(columns="Time (ms)").min(axis=0)
    return pd.DataFrame({
        "Stimulus": peak_values.index,
        "Peak (nA or mV)": peak_values.values
    })


def plot_iv_curve(iv_df, steady_col="SteadyState (nA or mV)", peak_col="Peak (nA or mV)"):
    """
    Plot IV curves from a DataFrame with steady-state and/or peak measurements.

    Parameters:
        iv_df (pd.DataFrame): Must include 'Stimulus' and the two value columns.
        steady_col (str): Column name for steady-state values.
        peak_col (str): Column name for peak values.
    """
    # Convert Stimulus labels to numeric if needed (e.g., "50 pA" → 50)
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
