# fit_simulation.py

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from neuron import h
import MNTB_PN_myFunctions as mFun
from MNTB_PN_fit import MNTB

h.load_file("stdrun.hoc")
h.celsius = 35

def load_fitted_params(results_dir):
    path = os.path.join(results_dir, "all_fitted_params.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Active parameters not found: {path}")
    df = pd.read_csv(path)
    return df.loc[0].to_dict()

def run_full_simulation(results_dir, age):
    print("🔬 Running final current-clamp simulation...")
    timestamp = os.path.basename(results_dir).split("_")[-1]

    totalcap = 25  # pF
    somaarea = (totalcap * 1e-6) / 1  # cm²
    ek = -106.81
    ena = 62.77

    params = load_fitted_params(results_dir)
    gleak = params["gleak"]
    gklt  = params["gklt"]
    gh    = params["gh"]
    erev  = params["erev"]
    gka   = params["gka"]
    gna   = params["gna"]
    gkht  = params["gkht"]
    stim_amp_fit = params["stim_amp"]

    cam = params.get("cam", 100)
    kam = params.get("kam", 0.037)
    cbm = params.get("cbm", 1)
    kbm = params.get("kbm", -0.043)
    cah = params.get("cah", 0.000533)
    kah = params.get("kah", -0.0909)
    cbh = params.get("cbh", 0.787)
    kbh = params.get("kbh", 0.0691)
    can = params.get("can", 0.2719)
    kan = params.get("kan", 0.04)
    cbn = params.get("cbn", 0.1974)
    kbn = params.get("kbn", 0.0)
    cap = params.get("cap", 0.00713)
    kap = params.get("kap", -0.1942)
    cbp = params.get("cbp", 0.0935)
    kbp = params.get("kbp", 0.0058)

    amps = np.round(np.arange(-0.100, 0.4, 0.010), 3)
    stimdelay, stimdur, totalrun = 10, 300, 510
    v_init = -77

    t_min = stimdelay + stimdur - 60
    t_max = stimdelay + stimdur - 10

    my_cell = MNTB(0, somaarea, erev, gleak, ena, gna, gh, gklt, gkht, gka, ek,
                   cam, kam, cbm, kbm, cah, kah, cbh, kbh, can, kan,
                   cbn, kbn, cap, kap, cbp, kbp)

    stim = h.IClamp(my_cell.soma(0.5))
    stim_traces = h.Vector().record(stim._ref_i)
    soma_v = h.Vector().record(my_cell.soma(0.5)._ref_v)
    t = h.Vector().record(h._ref_t)

    ap_counts, average_soma_values = [], []
    fig1, ax1 = plt.subplots()

    for amp in amps:
        mFun.custom_init(v_init)
        soma_vals, stim_vals, t_vals = mFun.run_simulation(amp, stim, soma_v, t, totalrun, stimdelay, stimdur, stim_traces)
        soma_vals_range, t_vals_range, average_soma_values = mFun.avg_ss_values(soma_vals, t_vals, t_min, t_max, average_soma_values)
        num_spikes, _, ap_counts, _ = mFun.count_spikes([], stimdelay, stimdur, h.Vector(), ap_counts, [])
        ax1.plot(t_vals, soma_vals, linewidth=0.5)

    # Calculate RMP and Rin
    rmp = [v for a, v in zip(amps, average_soma_values) if a == 0][0]
    Rin = calculate_input_resistance_between_minus20_plus20(amps, average_soma_values)

    print(f"🧪 RMP = {rmp:.2f} mV | Rin = {Rin:.3f} GΩ")
    print(f"📉 Saving summary data to: {results_dir}")

    annotate_params(ax1, rmp, Rin, gleak, gna, gh, gklt, gkht, gka, erev, ek, ena)
    fig1.savefig(os.path.join(results_dir, "trace_voltage_all_currents.png"), dpi=300)

    save_summary_data(results_dir, amps, average_soma_values, ap_counts, timestamp)

def annotate_params(ax, rmp, Rin, gleak, gna, gh, gklt, gkht, gka, erev, ek, ena):
    text = f"""RMP: {rmp:.2f} mV
Rin: {Rin:.3f} GΩ
gLeak: {gleak:.2f} nS
gNa: {gna:.2f} nS
gIH: {gh:.2f} nS
gKLT: {gklt:.2f} nS
gKHT: {gkht:.2f} nS
gKA: {gka:.2f} nS
ELeak: {erev:.2f} mV
Ek: {ek:.2f} mV
ENa: {ena:.2f} mV"""
    ax.annotate(
        text, xy=(100, -80), xytext=(300, -50),
        fontsize=10, bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="lightyellow")
    )

def calculate_input_resistance_between_minus20_plus20(amps, voltages):
    i1, i2 = np.argmin(np.abs(amps + 0.02)), np.argmin(np.abs(amps - 0.02))
    v1, v2 = voltages[i1], voltages[i2]
    delta_v, delta_i = v2 - v1, amps[i2] - amps[i1]
    return (delta_v / delta_i) / 1000 if abs(delta_i) > 1e-9 else None

def save_summary_data(results_dir, amps, avg_v, ap_counts, timestamp):
    df = pd.DataFrame({
        "Stimulus_nA": amps,
        "SteadyStateVoltage_mV": avg_v,
        "AP_count": ap_counts
    })
    df.to_csv(os.path.join(results_dir, f"summary_data_{timestamp}.csv"), index=False)
    print("✅ Simulation summary saved.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--age", type=str, default="P9")
    args = parser.parse_args()

    run_full_simulation(args.results_dir, args.age)

if __name__ == "__main__":
    main()
