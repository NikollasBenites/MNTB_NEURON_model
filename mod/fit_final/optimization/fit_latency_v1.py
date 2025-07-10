"""
fit_latency.py

Fit and optimize ion channel conductances (gKLT, gH, gLeak, E_leak, etc.)
in a single-compartment NEURON model to reproduce experimentally measured
latency to threshold from current-clamp recordings.

Latency is extracted using the dV/dt method (threshold crossing), and
fitting is performed using explained sum of squares (ESS) minimization.

The script:
- Parses filename metadata (e.g., age, phenotype, stimulus cap)
- Loads experimental latency vs. stimulus data (skips first evoked spike)
- Simulates latency in NEURON for each current injection
- Minimizes the difference between experimental and simulated latencies
- Outputs best-fit parameters, fit quality metrics, and latency model predictions
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['pdf.fonttype'] = 42   # TrueType
from scipy.optimize import minimize, differential_evolution
from scipy.signal import find_peaks
from sklearn.metrics import mean_squared_error, r2_score
from collections import namedtuple
from neuron import h
import MNTB_PN_myFunctions as mFun
from MNTB_PN_fit import MNTB

import time
import datetime
import json
import sys

# --- Named tuple to return latency parameters
refinedParams = namedtuple("refinedParams", ["gleak", "gklt", "gh", "erev", "gkht", "gna", "gka"])
def nstomho(x, somaarea):
    return (1e-9 * x / somaarea)  # Convert nS to mho/cm²


def fit_latency(filename,param_file):
    start_time = time.time()
    # --- Load experimental data
    file_base = os.path.splitext(os.path.basename(filename))[0]
    # === Extract age from filename
    age_str = "P9"
    for part in file_base.split("_"):
        if part.startswith("P") and part[1:].isdigit():
            age_str = part
            break
    try:
        age = int(age_str[1:])
    except:
        age = 0

    # === Extract phenotype (group)
    if "TeNT" in file_base:
        phenotype = "TeNT"
    elif "iMNTB" in file_base:
        phenotype = "iMNTB"
    else:
        phenotype = "WT"
    print(f"📌 Detected age: {age_str} (P{age}), Phenotype: {phenotype}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "..", "data","fit_passive", "fit_passive",filename)
    experimental_data = pd.read_csv(data_path)
    param_file = os.path.join(script_dir, "..", "results", "_fit_results", "_latest_all_fitted_params", f"{phenotype}",
                              param_file)
    if not os.path.exists(param_file):
        raise FileNotFoundError(f"Passive parameters not found at: {param_file}")
    param_df = pd.read_csv(param_file)
    # Extract the first row
    gleak = param_df.loc[0, "gleak"]
    gklt = param_df.loc[0, "gklt"]
    gh = param_df.loc[0, "gh"]
    erev = param_df.loc[0, "erev"]
    gkht = param_df.loc[0, "gkht"]
    gna = param_df.loc[0, "gna"]
    gka = param_df.loc[0, "gka"]

    print(f"Loaded parameters: gleak={gleak}, gklt={gklt}, gh={gh}, erev={erev}, gkht={gkht}, gna={gna}, gka={gka}")

    lat_thres_col = "Latency to Threshold (ms)"
    lat_peak_col = "Latency to Peak (ms)"
    stim_col = experimental_data["Stimulus (pA)"].values * 1e-3  # pA to nA
    non_null_latency = experimental_data[experimental_data[lat_thres_col].notna()]
    if not non_null_latency.empty:
        rheobase = non_null_latency.iloc[0]["Stimulus (pA)"] * 1e-3
    lat_values = experimental_data[experimental_data[lat_thres_col].notna()].iloc[2:] #starting 40pA more than Rheobase
    fit_currents = lat_values["Stimulus (pA)"].values * 1e-3
    fit_lat_thresh = lat_values[lat_thres_col].values  # ms
    fit_lat_peak = lat_values[lat_peak_col].values #ms

    # --- NEURON setup
    h.load_file("stdrun.hoc")
    h.celsius = 35
    h.dt = 0.02
    v_init = -70

    totalcap = 25  # pF
    somaarea = (totalcap * 1e-6) / 1  # cm²
    ek = -106.81
    ena = 62.77

    ################# sodium kinetics
    cam = 76.4 #76.4
    kam = .037
    cbm = 6.930852 #6.930852
    kbm = -.043

    cah = 0.000533  #( / ms)
    kah = -0.0909   #( / mV)
    cbh = 0.787     #( / ms)
    kbh = 0.0691    #( / mV)

    relaxation = 200

    def run_simulation(p: refinedParams,stim_amp, stim_dur=300, stim_delay=10):
        """
        Run simulation with 200 ms internal relaxation before stimulus.
        Returns the full 710 ms trace, with real stimulus at 210 ms.
        """
        v_init = -75
        totalcap = 25  # pF
        somaarea = (totalcap * 1e-6) / 1  # cm² assuming 1 µF/cm²

        param_dict = {
            "gna": p.gna,
            "gkht": p.gkht,
            "gklt": p.gklt,
            "gh": p.gh,
            "gka": p.gka,
            "gleak": p.gleak,
            "cam": cam,
            "kam": kam,
            "cbm": cbm,
            "kbm": kbm,
            "cah": cah,
            "kah": kah,
            "cbh": cbh,
            "kbh": kbh,
            "erev": erev,
            "ena": ena,
            "ek": ek,
            "somaarea": somaarea
        }

        # Simulate with 200 ms relaxation offset
        t, v = mFun.run_unified_simulation(
            MNTB_class=MNTB,
            param_dict=param_dict,
            stim_amp=stim_amp,
            stim_delay=stim_delay + relaxation,  # ms extra relaxation
            stim_dur=stim_dur,
            v_init=v_init,
            total_duration=510 + relaxation,  # full  ms sim
            return_stim=False
        )

        return t, v

    def compute_ess(params):
        gleak, gklt, gh, erev, gkht, gna, gka = params

        sim_lat_thresh = []
        sim_lat_peak = []

        # --- Rheobase penalty ---
        p_rheo = refinedParams(gleak, gklt, gh, erev, gkht, gna, gka)
        # p_rheo = p_rheo._replace(stim_amp=rheobase)
        t_rheo, v_rheo = run_simulation(p_rheo, stim_amp=rheobase)

        dvdt_rheo = np.gradient(v_rheo, t_rheo)
        stim_start_idx = np.where(t_rheo >= 210.5)[0][0]  # stim delay = 10 + relaxation
        above_thresh = np.where(dvdt_rheo[stim_start_idx:] > 45)[0]
        penalty = 0
        if len(above_thresh) == 0:
            penalty += 1000

        # --- Simulate suprathreshold steps ---
        for current in fit_currents:
            p = refinedParams(gleak, gklt, gh, erev, gkht, gna, gka)
            t, v = run_simulation(p, stim_amp=current)

            dvdt = np.gradient(v, t)
            stim_start_idx = np.where(t >= 210)[0][0]  # stim delay = 10 + relaxation
            time_segment = t[stim_start_idx:]
            dvdt_segment = dvdt[stim_start_idx:]
            voltage_segment = v[stim_start_idx:]

            # Latency to threshold
            above_thresh = np.where(dvdt_segment > 35)[0]
            if len(above_thresh) > 0:
                latency_thresh = time_segment[above_thresh[0]] - 210
            else:
                latency_thresh = np.nan

            # Latency to peak
            if len(voltage_segment) > 0:
                peak_idx = np.argmax(voltage_segment)
                latency_peak = time_segment[peak_idx] - 210
            else:
                latency_peak = np.nan

            sim_lat_thresh.append(latency_thresh)
            sim_lat_peak.append(latency_peak)

        sim_lat_thresh = np.array(sim_lat_thresh)
        sim_lat_peak = np.array(sim_lat_peak)

        valid_thresh = ~np.isnan(fit_lat_thresh) & ~np.isnan(sim_lat_thresh)
        valid_peak = ~np.isnan(fit_lat_peak) & ~np.isnan(sim_lat_peak)

        ess_thresh = np.sum((fit_lat_thresh[valid_thresh] - sim_lat_thresh[valid_thresh]) ** 2)
        ess_peak = np.sum((fit_lat_peak[valid_peak] - sim_lat_peak[valid_peak]) ** 2)

        total_ess = ess_thresh + ess_peak + penalty
        return total_ess

    # --- Initial guess and bounds
    print(f"These are the conductance values:"
                 
          f"\n gleak {gleak}"
          f"\n gklt {gklt}"
          f"\n gh {gh}"
          f"\n erev {erev}"
          f"\n gkht {gkht}"
          f"\n gna {gna}"
          f"\n gka {gka}")

       # --- Initial guess
    initial = [gleak, gklt, gh, erev, gkht, gna, gka]

    lbgNa = 0.9
    hbgNa = 1.1

    lbKht = 0.8
    hbKht = 1.2

    lbKlt = 0.8
    hbKlt = 1.2

    lbih = 0.9
    hbih = 1.1

    lbleak = 0.9
    hbleak = 1.1

    lbka = 0.1
    hbka = 1.9

    bounds = [
        (gleak * lbleak, gleak * hbleak),  # gleak
        (gklt * lbKlt, gklt * hbKlt),  # gklt
        (gh * lbih, gh * hbih),  # gh
        (erev, erev),  # fixed erev
        (gkht * lbKht, gkht * hbKht),  # gkht
        (gna * lbgNa, gna * hbgNa),  # gna
        (gka * lbka, gka * hbka)  # gka
    ]
    print("🔍 Running latency optimization...")

    result = minimize(
        compute_ess,
        x0=initial,
        bounds=bounds,
        method='L-BFGS-B',
        options={
            'disp': True,
            'maxiter': 10000,
            'ftol': 1e-6
        }
    )

    best_params = result.x
    final_cost = result.fun

    print(f"\n✅ Optimization complete. Final ESS = {final_cost:.3f}")
    print("Best-fit params:")
    for name, val in zip(refinedParams._fields, best_params):
        print(f"  {name}: {val:.6f}")
    # result = differential_evolution(
    #     compute_ess,
    #     bounds=bounds,
    #     strategy='best1bin',
    #     popsize=15,
    #     tol=1e-4,
    #     mutation=(0.5, 1),
    #     recombination=0.7,
    #     polish=True,
    #     disp=True
    # )
    # best_params = result.x
    # final_cost = result.fun
    # print(f"\n✅ Optimization complete. Final ESS = {final_cost:.3f}")
    # print("Best-fit params:")
    # for name, val in zip(refinedParams._fields, best_params):
    #     print(f"  {name}: {val:.6f}")

    # === Recompute simulated latencies using best-fit parameters ===
    sim_lat_thresh = []
    for current in fit_currents:
        p = refinedParams(*best_params)
        t, v = run_simulation(p, stim_amp=current)

        dvdt = np.gradient(v, t)
        stim_start_idx = np.where(t >= 210)[0][0]
        time_segment = t[stim_start_idx:]
        dvdt_segment = dvdt[stim_start_idx:]

        above_thresh = np.where(dvdt_segment > 35)[0]
        if len(above_thresh) > 0:
            latency = time_segment[above_thresh[0]] - 210
        else:
            latency = np.nan

        sim_lat_thresh.append(latency)

    # === Convert currents back to pA for readability ===
    currents_pA = fit_currents * 1e3

    # === Plot ===
    plt.figure(figsize=(6, 4))
    plt.plot(currents_pA, fit_lat_thresh, 'o-', label='Experimental', linewidth=2)
    plt.plot(currents_pA, sim_lat_thresh, 's--', label='Simulated (fit)', linewidth=2)
    plt.xlabel("Stimulus (pA)")
    plt.ylabel("Latency to Threshold (ms)")
    plt.title("Latency Fit: Experimental vs. Simulated")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # === Plot simulated voltage traces for a few currents ===
    example_currents = [stim_col]  # in nA
    plt.figure(figsize=(8, 6))

    for current in example_currents:
        p = refinedParams(*best_params)
        t, v = run_simulation(p, stim_amp=current)
        plt.plot(t, v, label=f"{int(current * 1e3)} pA")

    plt.axvline(210, color='gray', linestyle='--', label='Stim Start')
    plt.xlabel("Time (ms)")
    plt.ylabel("Membrane Potential (mV)")
    plt.title("Simulated Voltage Traces")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --- Apply best-fit params
    soma.g_leak = nstomho(opt_leak, somaarea)
    soma.gkltbar_LT_dth = nstomho(opt_gklt, somaarea)
    soma.ghbar_IH_nmb = nstomho(opt_gh, somaarea)
    soma.erev_leak = opt_erev
    soma.gkhtbar_HT_dth_nmb = nstomho(opt_gkht, somaarea)
    soma.gnabar_NaCh_nmb = nstomho(opt_gna, somaarea)
    soma.gkabar_ka = nstomho(opt_gka, somaarea)

    simulated_voltages_full = np.array([run_simulation(i) for i in stim_col])

    # --- Compute simulated voltages only for the fitted currents
    sim_fit = np.array([run_simulation(i) for i in fit_currents])

    # --- Fit-specific RMSE and R²
    rmse_fit = np.sqrt(mean_squared_error(fit_voltages, sim_fit))
    r2_fit = r2_score(fit_voltages, sim_fit)

    # --- Full-trace (all) RMSE and R² (already present)
    rmse_all = np.sqrt(mean_squared_error(all_steady_state_voltages, simulated_voltages_full))
    r2_all = r2_score(all_steady_state_voltages, simulated_voltages_full)

    residuals = all_steady_state_voltages - simulated_voltages_full
    rmse = np.sqrt(mean_squared_error(all_steady_state_voltages, simulated_voltages_full))
    r2 = r2_score(all_steady_state_voltages, simulated_voltages_full)

    # --- Save outputs
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(script_dir, "..", "figures", f"fit_passive_{file_base}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    param_txt = os.path.join(script_dir, "..", "results", "_fit_results", f"passive_params_{file_base}_{timestamp}.txt")
    with open(param_txt, "w") as f:
        f.write(f"{opt_leak},{opt_gklt},{opt_gh},{opt_erev},{opt_gkht},{opt_gna},{opt_gka}\n")

    # legacy support
    with open(os.path.join(script_dir, "..", "results", "_fit_results", "best_fit_params.txt"), "w") as f:
        f.write(f"{opt_leak},{opt_gklt},{opt_gh},{opt_erev},{opt_gkht},{opt_gna},{opt_gka}\n")

    # --- Input resistance
    mask = (stim_col >= -0.020) & (stim_col <= 0.020)
    rin_exp = np.polyfit(stim_col[mask], all_steady_state_voltages[mask], 1)[0]
    rin_sim = np.polyfit(stim_col[mask], simulated_voltages_full[mask], 1)[0]

    # --- Summary JSON
    summary_path = os.path.join(output_dir, f"passive_summary_{file_base}.json")
    summary = {
        "gleak": opt_leak, "gklt": opt_gklt, "gh": opt_gh, "erev": opt_erev,
        "gkht": opt_gkht, "gna": opt_gna, "gka": opt_gka,
        "rin_exp_mohm": rin_exp,
        "rin_sim_mohm": rin_sim,
        "rmse_fit_mV": rmse_fit,
        "r2_fit": r2_fit,
        "rmse_all_mV": rmse_all,
        "r2_all": r2_all
    }

    print("\n📈 Fit Quality Metrics:")
    print(f"RMSE (fit points only): {rmse_fit:.2f} mV")
    print(f"R²   (fit points only): {r2_fit:.4f}")
    print(f"RMSE (all points):      {rmse_all:.2f} mV")
    print(f"R²   (all points):      {r2_all:.4f}")

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)

    # --- Plot VI curve
    x_min = -0.15
    x_max = 0.4
    y_min = -110
    y_max = -20
    plt.figure(figsize=(8, 8))
    plt.scatter(stim_col, all_steady_state_voltages, color='r', label="Experimental")
    plt.plot(stim_col, simulated_voltages_full, '-', color='b', label="Simulated")
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel("Injected Current (nA)")
    plt.ylabel("Steady-State Voltage (mV)")
    plt.title(f"Passive Fit: Experimental vs Simulated {file_base}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "passive_fit.pdf"), dpi=300)
    plt.close()

    # --- Plot Input Resistance
    plt.figure(figsize=(8, 8))
    plt.plot(stim_col[mask], all_steady_state_voltages[mask], 'o', label="Exp")
    plt.plot(stim_col[mask], rin_exp * stim_col[mask] + np.mean(all_steady_state_voltages[mask]), '-', label=f"Exp Fit ({rin_exp:.2f} MΩ)")
    plt.plot(stim_col[mask], simulated_voltages_full[mask], 's', label="Sim")
    plt.plot(stim_col[mask], rin_sim * stim_col[mask] + np.mean(simulated_voltages_full[mask]), '--', label=f"Sim Fit ({rin_sim:.2f} MΩ)")
    plt.xlabel("Injected Current (nA)")
    plt.ylabel("Steady-State Voltage (mV)")
    plt.title("Input Resistance Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "input_resistance_fit.png"), dpi=300)
    plt.close()

    # --- Plot Residual
    plt.figure(figsize=(8, 8))
    plt.bar(stim_col, residuals, width=0.01, color='purple', alpha=0.7)
    plt.axhline(0, color='gray', linestyle='--')
    plt.xlabel("Injected Current (nA)")
    plt.ylabel("Residual Voltage (mV)")
    plt.title("Residuals: Experimental - Simulated")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "residuals.pdf"), dpi=300)
    plt.close()

    print(f"✅ Passive fit complete: results saved to {output_dir}")
    print(f"⏱️ Elapsed: {time.time() - start_time:.2f} s")

    return refinedParams(opt_leak, opt_gklt, opt_gh, opt_erev, opt_gkht, opt_gna, opt_gka), output_dir


# === Optional CLI ===
if __name__ == "__main__":
    import argparse

    # === DEBUG MODE SIMULATION ===
    if sys.gettrace():  # This returns True when running in debugger
        sys.argv = [
            sys.argv[0],  # the script name
            "--data", "/Users/nikollas/Library/CloudStorage/OneDrive-UniversityofSouthFlorida/MNTB_neuron/mod/fit_final/data/latency_results/latency_data_02062024_P9_FVB_PunTeTx_TeNT_S4C1.csv",
            "--param_file","/Users/nikollas/Library/CloudStorage/OneDrive-UniversityofSouthFlorida/MNTB_neuron/mod/fit_final/results/_fit_results/_latest_all_fitted_params/all_fitted_params_sweep_13_clipped_510ms_02072024_P9_FVB_PunTeTx_Dan_iMNTB_160pA_S3C3_20250624_154105.csv"# your arguments
        ]
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to experimental CSV with columns 'Current', 'SteadyStateVoltage'")
    parser.add_argument("--param_file", required=True, help="Path to parameter file")
    args = parser.parse_args()
    fit_latency(args.data,args.param_file)
