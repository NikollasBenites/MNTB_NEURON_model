import os
from neuron import h
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, minimize
import time
np.random.seed(1)
start = time.time()
# ========== Utilities ==========

def nstomho(x, somaarea):
    return (1e-9 * x / somaarea)  # Convert nS to mho/cm²

def setup_model():
    totalcap = 20  # pF
    somaarea = (totalcap * 1e-6) / 1  # cm^2

    h.load_file("stdrun.hoc")
    soma = h.Section(name='soma')
    soma.L = 15
    soma.diam = 15
    soma.Ra = 150
    soma.cm = 1
    soma.v = -70

    soma.insert('leak')
    soma.insert('LT')   # Kv1
    soma.insert('IH')   # HCN
    soma.insert('HT')   # Kv3
    soma.insert('NaCh') # Na

    soma.gkhtbar_HT = nstomho(300, somaarea)
    soma.gnabar_NaCh_nmb = nstomho(300, somaarea)

    soma.ek = -106.8
    soma.ena = 62.77

    return soma, somaarea

def simulate_current_response(soma, somaarea, st, current, gleak, gklt, gh, erev):
    soma.g_leak = nstomho(gleak, somaarea)
    soma.gkltbar_LT = nstomho(gklt, somaarea)
    soma.ghbar_IH = nstomho(gh, somaarea)
    soma.erev_leak = erev

    st.amp = current

    v_vec = h.Vector()
    t_vec = h.Vector()
    v_vec.record(soma(0.5)._ref_v)
    t_vec.record(h._ref_t)

    h.finitialize(-70)
    h.run()

    t = np.array(t_vec)
    v = np.array(v_vec)
    mask = (t >= 250) & (t <= 300)
    return np.mean(v[mask])

# ========== Objective Function ==========

def compute_ess(params, exp_currents, exp_voltages):
    gleak, gklt, gh, erev = params
    soma, somaarea = setup_model()

    st = h.IClamp(soma(0.5))
    st.dur = 300
    st.delay = 10
    h.tstop = 510

    simulated = []
    for current in exp_currents:
        ss_voltage = simulate_current_response(soma, somaarea, st, current, gleak, gklt, gh, erev)
        simulated.append(ss_voltage)

    simulated = np.array(simulated)
    ess = np.sum((exp_voltages - simulated) ** 2)
    return ess

# ========== Main ==========

def main():
    # Load experimental data
    df = pd.read_csv("/Users/nikollas/Library/CloudStorage/OneDrive-UniversityofSouthFlorida/MNTB_neuron/MNTB_Model_Dann/experimental_data_P9.csv")
    exp_currents = df["Current"].values * 1e-3  # pA to nA
    exp_voltages = df["SteadyStateVoltage"].values

    # Bounds for [gleak, gklt, gh, erev]
    bounds = [(0, 20), (0, 200), (0, 50), (-90, -50)]

    print("⏳ Running global search with differential_evolution...")
    result_de = differential_evolution(
        compute_ess,
        bounds,
        args=(exp_currents, exp_voltages),
        updating="deferred",
        workers=-1,  # Use all available CPU cores
        polish=False
    )

    print(f" DE result: {result_de.fun:.4f} at {result_de.x}")
    mid = time.time()
    print(f"🌍 DE took {mid - start:.2f} seconds")
    print("🔍 Refining with local minimize...")
    result_min = minimize(
        compute_ess,
        result_de.x,
        args=(exp_currents, exp_voltages),
        bounds=bounds
    )

    optimal_leak, optimal_gklt, optimal_gh, optimal_erev = result_min.x
    print(f" Final result: ESS = {result_min.fun:.4f}")
    print(f"Optimal parameters:\n"
          f"  g_leak = {optimal_leak:.3f} nS\n"
          f"  gklt = {optimal_gklt:.3f} nS\n"
          f"  gh = {optimal_gh:.3f} nS\n"
          f"  erev = {optimal_erev:.3f} mV")

    # Simulate best-fit trace
    soma, somaarea = setup_model()
    st = h.IClamp(soma(0.5))
    st.dur = 300
    st.delay = 10
    h.tstop = 510

    best_fit_voltages = []
    for current in exp_currents:
        ss_voltage = simulate_current_response(soma, somaarea, st, current, optimal_leak, optimal_gklt, optimal_gh, optimal_erev)
        best_fit_voltages.append(ss_voltage)

    end = time.time()
    print(f"🔍 minimize() took {end - mid:.2f} seconds")
    print(f"🕒 Total time: {end - start:.2f} seconds")

    # Plot results
    plt.figure(figsize=(12, 7))
    plt.scatter(exp_currents, exp_voltages, color='red', label='Experimental Data')
    plt.plot(exp_currents, best_fit_voltages, 'o-', color='blue', label='Best-Fit Simulation')
    plt.xlabel("Injected Current (nA)", fontsize=16)
    plt.ylabel("Steady-State Voltage (mV)", fontsize=16)
    plt.title("Experimental vs Simulated Steady-State Voltage", fontsize=16)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()



# ========== Entry Point Guard ==========

if __name__ == "__main__":
    main()
