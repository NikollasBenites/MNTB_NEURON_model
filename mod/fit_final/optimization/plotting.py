import pandas as pd
import matplotlib.pyplot as plt
import os
from os.path import split
import numpy as np
filename = ("04092024_P4_FVB_PunTeTx_Dan.dat").split(".")[0]
# filename = "all_sweeps_12172022_P9_FVB_PunTeTx.csv"
exp = "simulation" #the experiment type
script_dir = os.path.dirname(os.path.abspath(__file__))
sim_path = os.path.join(script_dir, "..", "data","exported_sweeps")
sim_dirs = [f for f in os.listdir(sim_path) if f.startswith(f"{exp}_{filename}")]
# sim_dirs = "/Users/nikollas/Library/CloudStorage/OneDrive-UniversityofSouthFlorida/MNTB_neuron/mod/fit_final/data/exported_sweeps/"

def search_file():
    if sim_dirs:
        latest_folder = max(sim_dirs)
        voltage_traces = os.path.join(sim_path, latest_folder, "voltage_traces.csv")
        # voltage_traces = os.path.join("/Users/nikollas/Library/CloudStorage/OneDrive-UniversityofSouthFlorida/MNTB_neuron/mod/fit_final/data/exported_sweeps/all_sweeps_12172022_P9_FVB_PunTeTx.csv")
        if os.path.exists(voltage_traces):
            df_voltage = pd.read_csv(voltage_traces)
            print(f"Found voltage traces in {voltage_traces}")
            return df_voltage
        else:
            print("file does not exist.")


def plot_voltage_traces(df_voltage, title="Voltage Traces", save_fig=False, dpi=300):
    """
    Plots voltage traces from a DataFrame.

    Parameters:
    - df_voltage: pd.DataFrame with time in the first column, and voltage traces in remaining columns.
    - title: Optional title for the plot.
    """
    # if df_voltage.columns[1].startswith("Sweep"):
    #     n_sweeps = df_voltage.shape[1] - 1
    #     amps = np.round(np.arange(-0.1, -0.1 + 0.01 * n_sweeps, 0.01), 3)
    #     amp_labels = [f"{amp} nA" for amp in amps]
    #     df_voltage.columns = [df_voltage.columns[0]] + amp_labels

    if df_voltage is None or df_voltage.empty:
        print("DataFrame is empty or None. Nothing to plot.")
        return
    print("df dtypes:")
    print(df_voltage.dtypes)
    # Convert all columns except Time to numeric (in-place, handles strings)
    for col in df_voltage.columns[1:]:
        df_voltage[col] = pd.to_numeric(df_voltage[col], errors='coerce')

    plt.figure(figsize=(10, 6))
    time = df_voltage.iloc[:, 0]

    spike_threshold = -20  # mV
    colors = ['black', 'lightgray']

    # Step 1: Find rheobase trace from all sweeps (only check up to 310 ms)
    rheobase_col = None
    time_limit = 310  # ms
    time_mask = time <= time_limit

    for col in df_voltage.columns[1:]:
        trace = df_voltage[col]
        if (trace[time_mask] > spike_threshold).any():
            rheobase_col = col
            print(f"Rheobase detected in: {col} (before {time_limit} ms)")
            break

        # Step 2: Get column list in 20pA step
    sweep_cols = df_voltage.columns[1::2].tolist()

    # Step 3: Include rheobase if it was skipped
    if rheobase_col and rheobase_col not in sweep_cols:
        sweep_cols.append(rheobase_col)

    # Step 4: Sort by stimulus order (based on column names)
    sweep_cols_sorted = sorted(sweep_cols, key=lambda x: float(x.split()[0]))

    # Step 5: Plot only up to and including rheobase
    for i, col in enumerate(sweep_cols_sorted):
        trace = df_voltage[col]
        color = 'black' if col == rheobase_col else colors[i % 2]
        plt.plot(time, trace, color=color, linewidth=1)

        if col == rheobase_col:
            break  # ✅ stop after rheobase


    plt.xlabel("Time (ms)")
    plt.ylabel("Voltage (mV)")
    #plt.title(title)
    if len(df_voltage.columns[1:]) <= 10:  # Avoid messy legend with too many traces
        plt.legend(loc="upper right", fontsize="small", ncol=2)
    x_scale = 50  # 10 ms
    y_scale = 20  # 20 mV

    xlim = plt.xlim()
    ylim = plt.ylim()

    # Position (adjust as needed)
    x_start = xlim[1] - 2.51 * x_scale
    y_start = ylim[0] + 0.08 * (ylim[1] - ylim[0])

    # Draw horizontal (time) scale bar
    plt.hlines(y=y_start, xmin=x_start, xmax=x_start + x_scale, linewidth=2, color='black')
    plt.text(x_start + x_scale / 2, y_start - 0.03 * (ylim[1] - ylim[0]), f"{x_scale} ms",
             ha='center', va='top', fontsize=14)

    # Draw vertical (voltage) scale bar
    plt.vlines(x=x_start, ymin=y_start, ymax=y_start + y_scale, linewidth=2, color='black')
    plt.text(x_start - 0.01 * (xlim[1] - xlim[0]), y_start + y_scale / 2, f"{y_scale} mV",
             ha='right', va='center', fontsize=14,rotation=90)

    plt.grid(False)
    plt.tight_layout()
    plt.axis('off')
    if save_fig:
        # Build the output path

        if sim_dirs:
            latest_dir = max(sim_dirs)
            base_filename = os.path.join(sim_path, latest_dir, "voltage_traces_plot")
            plt.savefig(f"{base_filename}.png", dpi=dpi, bbox_inches='tight')
            plt.savefig(f"{base_filename}.pdf", dpi=dpi, bbox_inches='tight')
            print(f"✅ Figures saved to:\n{base_filename}.png\n{base_filename}.pdf")
        else:
            print("❌ No matching simulation directory found. Plot not saved.")
    plt.show()

df_voltage = search_file()
plot_voltage_traces(df_voltage,save_fig=True)


