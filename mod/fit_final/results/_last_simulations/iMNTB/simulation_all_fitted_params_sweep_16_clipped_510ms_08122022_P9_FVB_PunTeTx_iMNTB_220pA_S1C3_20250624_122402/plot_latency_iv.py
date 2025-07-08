import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from iv_analysis import (
    average_steady_state_iv,
    peak_current_iv,
    plot_iv_curve,
    latency_iv,
    latency_iv_dual,
)
from datetime import datetime
sweep_step = 20
# === Load data ===
file_path = os.path.abspath("voltage_traces.csv")  # Full path to input file
df = pd.read_csv(file_path)
df = df.rename(columns={"Time (s)": "Time (ms)"})

# === Define output directory and base filename ===
output_dir = os.path.dirname(file_path)
filename = "latency_iv"
date_stamp = datetime.now().strftime("%Y%m%d")  # Use only the date
save_basename = f"{filename}_{date_stamp}"

# === Run analysis ===
latency_df = latency_iv_dual(df, search_start_ms=10.5, search_end_ms=200, sweep_step=sweep_step,dvdt_threshold=35)

# === Plot Latency vs Current ===
plt.figure(figsize=(6, 5))
plt.plot(latency_df["Stimulus (pA)"], latency_df["Latency to Threshold (ms)"], 'o-', label="To Threshold")
plt.plot(latency_df["Stimulus (pA)"], latency_df["Latency to Peak (ms)"], 's--', label="To Peak")
plt.xlabel("Injected Current (pA)")
plt.ylabel("Latency (ms)")
plt.title("Latency vs. Current Injection")
plt.grid(True)
plt.legend()
plt.tight_layout()

# === Save figure and CSV ===
fig_path = os.path.join(output_dir, f"{save_basename}.png")
csv_path = os.path.join(output_dir, f"{save_basename}.csv")

plt.savefig(fig_path, dpi=300)
plt.close()

latency_df.to_csv(csv_path, index=False)

# === Status ===
print(f"✅ Saved figure to: {fig_path}")
print(f"✅ Saved CSV to: {csv_path}")

