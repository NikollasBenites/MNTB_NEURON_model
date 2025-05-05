# fit_passive.py

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from neuron import h
from scipy.optimize import minimize
from MNTB_PN_fit import MNTB, nstomho
import MNTB_PN_myFunctions as mFun
from datetime import datetime

h.load_file("stdrun.hoc")
np.random.seed(1)
h.celsius = 35

def load_experimental_data(age):
    data_path = os.path.join("data", f"experimental_data_{age}_TeNT_12232024_S1C1.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Experimental data not found: {data_path}")
    df = pd.read_csv(data_path)
    currents = df["Current"].values * 1e-3  # pA to nA
    voltages = df["SteadyStateVoltage"].values * 1000  # V to mV
    return currents, voltages

def create_soma():
    soma = h.Section(name='soma')
    soma.L = 20
    soma.diam = 15
    soma.Ra = 150
    soma.cm = 1
    soma.insert('leak')
    soma.insert('HT_dth')
    soma.insert('LT_dth')
    soma.insert('NaCh_nmb')
    soma.insert('IH_dth')
    soma.insert('ka')
    soma.ek = -106.8
    soma.ena = 62.77
    return soma

def run_passive_simulation(soma, stim, currents, v_init):
    v_vec, t_vec = h.Vector(), h.Vector()
    v_vec.record(soma(0.5)._ref_v)
    t_vec.record(h._ref_t)

    results = []
    for i in currents:
        stim.amp = i
        v_vec.resize(0)
        t_vec.resize(0)
        h.v_init = v_init
        mFun.custom_init(v_init)
        h.run()

        t_arr = np.array(t_vec)
        v_arr = np.array(v_vec)
        mask = (t_arr >= 250) & (t_arr <= 300)
        results.append(np.mean(v_arr[mask]))
    return np.array(results)

def compute_ess(params, soma, stim, currents, target_voltages, v_init, somaarea):
    gleak, gklt, gh, gka, erev, gkht, gna = params
    soma.g_leak = nstomho(gleak, somaarea)
    soma.gkltbar_LT_dth = nstomho(gklt, somaarea)
    soma.ghbar_IH_nmb = nstomho(gh, somaarea)
    soma.gkabar_ka = nstomho(gka, somaarea)
    soma.erev_leak = erev
    soma.gkhtbar_HT_dth = nstomho(gkht, somaarea)
    soma.gnabar_NaCh_nmb = nstomho(gna, somaarea)

    simulated = run_passive_simulation(soma, stim, currents, v_init)
    return np.sum((target_voltages - simulated) ** 2)

def fit_passive(results_dir, age):
    v_init = -70
    totalcap = 25  # pF
    somaarea = (totalcap * 1e-6) / 1  # cm²
    currents, voltages = load_experimental_data(age)
    soma = create_soma()
    stim = h.IClamp(soma(0.5))
    stim.dur = 300
    stim.delay = 10
    h.dt = 0.02
    h.tstop = stim.delay + stim.dur

    init = [15, 100, 25, 50, -50, 200, 200]
    bounds = [(0,30),(0,200),(0,50),(0,100),(-90,-30),(20,380),(20,380)]

    ess_fn = lambda params: compute_ess(params, soma, stim, currents, voltages, v_init, somaarea)
    result = minimize(ess_fn, init, bounds=bounds)
    gleak, gklt, gh, gka, erev, gkht, gna = result.x

    param_file = os.path.join(results_dir, "best_fit_params.txt")
    with open(param_file, "w") as f:
        f.write(f"{gleak},{gklt},{gh},{gka},{erev},{gkht},{gna}\n")
    print("✅ Passive parameters saved.")

    # Plot results
    simulated = run_passive_simulation(soma, stim, currents, v_init)
    plt.figure(figsize=(10, 6))
    plt.scatter(currents, voltages, color='red', label="Experimental")
    plt.plot(currents, simulated, 'o-', label="Best-fit")
    plt.xlabel("Current (nA)")
    plt.ylabel("Steady-State Voltage (mV)")
    plt.title("Passive Fit")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "passive_fit.png"), dpi=300)
    print("📊 Saved passive fit plot.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--file", type=str, default="P9")
    args = parser.parse_args()

    fit_passive(args.results_dir, args.age)

if __name__ == "__main__":
    main()
