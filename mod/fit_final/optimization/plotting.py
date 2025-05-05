import pandas as pd
import matplotlib.pyplot as plt
import os
from os.path import split
filename = ("12172022_P9_FVB_PunTeTx_phasic_iMNTB.dat").split(".")[0]
def search_file():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sim_path = os.path.join(script_dir,"..", "figures")
    sim_dirs = [f for f in os.listdir(sim_path) if f.startswith(f"simulation_{filename}")]
    if sim_dirs:
        latest_folder = max(sim_dirs)
        voltage_traces = os.path.join(sim_path, latest_folder, "voltage_traces.csv")
        if os.path.exists(voltage_traces):
            df_voltage = pd.read_csv(voltage_traces)
            print(f"Found voltage traces in {voltage_traces}")
            return df_voltage
        else:
            print("file does not exist.")


def plot_voltage_traces(df_voltage, title="Voltage Traces"):
    """
    Plots voltage traces from a DataFrame.

    Parameters:
    - df_voltage: pd.DataFrame with time in the first column, and voltage traces in remaining columns.
    - title: Optional title for the plot.
    """
    if df_voltage is None or df_voltage.empty:
        print("DataFrame is empty or None. Nothing to plot.")
        return
    print("df dtypes:")
    print(df_voltage.dtypes)

    plt.figure(figsize=(10, 6))
    time = df_voltage.iloc[:, 0]

    colors = ['black', 'gray']
    for i, col in enumerate(df_voltage.columns[1::2]):
            plt.plot(time, df_voltage[col],color=colors[i % 2],  linewidth=0.8)

    plt.xlabel("Time (ms)")
    plt.ylabel("Voltage (mV)")
    plt.title(title)
    if len(df_voltage.columns[1:]) <= 10:  # Avoid messy legend with too many traces
        plt.legend(loc="upper right", fontsize="small", ncol=2)
    plt.grid(False)
    plt.tight_layout()
    plt.show()

df_voltage = search_file()

plot_voltage_traces(df_voltage)


