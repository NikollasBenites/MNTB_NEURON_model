import os
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
folder_path = os.path.dirname(__file__)
print(f"The folder path is: {folder_path}")
csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

base_df =None
merged_df = None

for i, file in enumerate(csv_files):
    df = pd.read_csv(file)
    if df.shape[1] < 2:
        print(f"The file {file} has less than 2 columns")

    col1 = df.iloc[:,0]
    col2 = df.iloc[:,1]

    if base_df is None:
        merged_df = pd.DataFrame()
        merged_df["time"] = col1
        merged_df[os.path.basename(file)] = col2
        base_df = col1
    else:
        if not col1.equals(base_df):
            print(f"The file {file} has different columns")
            continue
        merged_df[os.path.basename(file)] = col2
print(merged_df.head())

time = merged_df["time"]
ap_traces = merged_df.drop(columns="time")

#plot ap traces
plt.figure(figsize=[10,8])
for col in ap_traces.columns:
    plt.plot(time, ap_traces[col], label=col)
plt.xlabel("Time (ms)")
plt.ylabel("Membrane potential (mV)")
plt.title("Overlay of Action Potentials")
plt.legend(loc='upper right', fontsize='small', frameon=False)
plt.tight_layout()
plt.show(block=False)

#Aligned by time
# === Average of unaligned APs ===
avg_trace = ap_traces.mean(axis=1)
std_trace = ap_traces.std(axis=1)

plt.figure(figsize=(10, 6))
plt.plot(time, avg_trace, color='black', label='Mean AP')
plt.fill_between(time, avg_trace - std_trace, avg_trace + std_trace, alpha=0.3, label='±1 SD')
plt.xlabel("Time (ms)")
plt.ylabel("Membrane potential (mV)")
plt.title("Average Action Potential (Unaligned)")
plt.legend()
plt.tight_layout()
plt.show(block=False)


time_ap = merged_df["time"].values
dt = time_ap[1] - time_ap[0]
ap_aligned = merged_df.drop(columns="time")

# === Parameters ===
dvdt_threshold = 20  # mV/ms
min_time = 11        # ms, start detecting threshold only after this time

aligned_time_traces = []
aligned_voltage_traces = []

for col in ap_aligned.columns:
    v = ap_aligned[col].values
    dvdt = np.gradient(v, dt)

    # Mask: only consider dv/dt after 10 ms
    mask_after_10ms = time_ap >= min_time
    dvdt_masked = dvdt[mask_after_10ms]
    time_masked = time_ap[mask_after_10ms]
    v_masked = v[mask_after_10ms]

    # Find threshold crossing after 10 ms
    thres_idx_rel = np.where(dvdt_masked > dvdt_threshold)[0]
    if len(thres_idx_rel) == 0:
        print(f"⚠️ No threshold in {col} after {min_time} ms, skipping.")
        continue

    thres_idx = np.where(mask_after_10ms)[0][0] + thres_idx_rel[0]  # map back to full index
    t_thresh = time_ap[thres_idx]
    v_thresh = v[thres_idx]

    # Align both time and voltage
    t_aligned = time_ap - t_thresh
    v_aligned = v - v_thresh

    aligned_time_traces.append(t_aligned)
    aligned_voltage_traces.append(v_aligned)


# === Plotting ===
plt.figure(figsize=(10, 8))
for t, v in zip(aligned_time_traces, aligned_voltage_traces):
    plt.plot(t, v, alpha=0.6)

plt.axvline(0, color='gray', linestyle='--', label="Threshold time")
plt.axhline(0, color='gray', linestyle=':', label="Threshold voltage")
plt.xlabel("Time from threshold (ms)")
plt.ylabel("Voltage from threshold (mV)")
plt.title("APs Aligned at Threshold (Time and Voltage)")
plt.legend()
plt.tight_layout()
plt.show(block=False)

#aligned by threshold
# === Build common time base ===
t_common = np.linspace(-10, 20, int((30)/dt))  # From -10 to +20 ms around threshold

# Interpolate all aligned traces onto t_common
aligned_matrix = []
for t, v in zip(aligned_time_traces, aligned_voltage_traces):
    interp = np.interp(t_common, t, v)
    aligned_matrix.append(interp)

aligned_matrix = np.array(aligned_matrix)
avg_aligned = aligned_matrix.mean(axis=0)
std_aligned = aligned_matrix.std(axis=0)

# === Plot average aligned trace ===
plt.figure(figsize=(10, 6))
plt.plot(t_common, avg_aligned, color='black', label='Mean Aligned AP')
plt.fill_between(t_common, avg_aligned - std_aligned, avg_aligned + std_aligned, alpha=0.3, label='±1 SD')
#plt.axvline(0, color='gray', linestyle='--', label="Threshold time")
#plt.axhline(0, color='gray', linestyle=':', label="Threshold voltage")
plt.xlabel("Time from threshold (ms)")
plt.ylabel("Voltage from threshold (mV)")
plt.title("Average Aligned Action Potential")
plt.legend()
plt.tight_layout()
plt.show()
