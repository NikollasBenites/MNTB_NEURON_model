import numpy as np
from matplotlib.pyplot import pause

np.random.seed(1)
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import minimize, differential_evolution
from neuron import h
import MNTB_PN_myFunctions as mFun
import time as timer
from load_heka_python.load_heka import LoadHeka
from multiprocessing import cpu_count
h.load_file('stdrun.hoc')
full_path_to_file = r"/Users/nikollas/Library/CloudStorage/OneDrive-UniversityofSouthFlorida/MNTB_neuron/MNTB_Model_Dann/10142022_P9_FVB_PunTeTx.dat"

with LoadHeka(full_path_to_file) as hf:
    hf.print_group_names()
    hf.print_series_names(group_idx=1)

    # Load series data
    series = hf.get_series_data(group_idx=1, series_idx=3, channel_idx=0, include_stim_protocol=True)
    voltage = series['data']
    time = series['time']
    stim = series.get('stim', None)
    if isinstance(stim, dict) and 'data' in stim:
        stim = stim['data']
    print("\n>>> stim type:", type(stim))
    print(">>> stim keys/shape/sample:")

    # Try the most common forms
    try:
        print("Keys:", list(stim.keys()))
    except Exception:
        pass

    try:
        print("Shape:", stim.shape)
    except Exception:
        pass

    try:
        print("First entry:", stim[0])
    except Exception:
        pass

    # Ensure stim is in list-of-arrays format
    if stim is not None:
        if isinstance(stim, np.ndarray):
            if stim.ndim == 1:
                # Single sweep, wrap into a list
                stim = [stim]
            elif stim.ndim == 2:
                # Already multiple sweeps: convert to list of arrays
                stim = [s for s in stim]
            else:
                raise ValueError("Unexpected stim shape:", stim.shape)
        else:
            # Catch any other types
            stim = list(stim)
    labels = series.get('labels', None)

    n_sweeps = len(voltage)
    # Filter out sweeps with non-numeric stim
    print(f"First 5 stim entries: {[stim[i] for i in range(min(5, len(stim)))]}")

    valid_sweeps = []
    for i in range(n_sweeps):
        try:
            _ = np.array(stim[i], dtype=float)
            valid_sweeps.append(i)
        except Exception:
            print(f"[WARN] Sweep {i} skipped due to non-numeric stim.")
        if len(valid_sweeps) == 0:
            raise RuntimeError("No valid sweeps found! All stim arrays were non-numeric or invalid.")

    # Check if labels is a valid list or array
    try:
        label_list = list(labels)
        if len(label_list) != n_sweeps:
            raise ValueError
    except:
        label_list = [None] * n_sweeps

    # Plot all sweeps
    plt.figure(figsize=(12, 6))
    for i in range(n_sweeps):
        stim_label = f"{label_list[i]} pA" if label_list[i] is not None else f"Sweep {i}"
        plt.plot(time[i] * 1000, voltage[i], label=f"Sweep {i}")

    plt.xlabel("Time (ms)")
    plt.ylabel("Membrane potential (mV)")
    plt.title("HEKA Sweeps - Inspect Before Fitting")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.grid(True)
    plt.tight_layout()
    plt.show(block=False)

    # Print sweep info
    print("\nAvailable sweeps:")
    for i in range(n_sweeps):
        stim_label = f"{label_list[i]} pA" if label_list[i] is not None else "unknown"
        print(f"  Sweep {i:2d} → {stim_label}")

    # # User selects sweep
    # sweep_idx = int(input(f"\nSelect sweep index (0 to {n_sweeps - 1}): "))
    # v_exp_heka = voltage[sweep_idx]
    # t_exp_heka = time[sweep_idx] * 1000


    # print(f"\nSelected Sweep {sweep_idx}: Length = {len(v_exp_heka)} samples")

    # Plot selected sweep
    # plt.figure()
    # plt.plot(t_exp_heka, v_exp_heka, color='black')
    # plt.xlabel("Time (ms)")
    # plt.ylabel("Voltage (mV)")
    # stim_label = f"{label_list[sweep_idx]} pA" if label_list[sweep_idx] is not None else "unknown"
    # plt.title(f"Sweep {sweep_idx} – {stim_label}")
    # plt.tight_layout()
    # plt.grid(True)
    # plt.show(block=False)

# Load experimental data
# experimentalTrace = np.genfromtxt('P9_iMNTB_Rheobase_raw.csv', delimiter=',', skip_header=1, dtype=float, filling_values=np.nan)
# t_exp = experimentalTrace[:,0]*1000  # ms
# t_exp = t_exp - t_exp[0]
# V_exp = experimentalTrace[:,2]  # mV
# V_exp = v_exp_heka*1000 #convert to mV
# t_exp = t_exp_heka
h.celsius = 35

#Create a dendrite
dend = h.Section(name='dend')
dend.diam = 3
dend.L = 150
dend.Ra = 150
dend.cm = 0.5
dend.insert('leak')

# Create soma section
soma = h.Section(name='soma')
soma.L = 15  # µm
soma.diam = 20  # µm
soma.Ra = 150
soma.cm = 1
v_init = -77
soma.insert('leak')
soma.insert('LT')
soma.insert('IH')
soma.insert('HT')
soma.insert('NaCh')
soma.ek = -106.8
soma.ena = 62.77

# Create axon section
axon = h.Section(name='axon')
axon.L = 15
axon.diam = 1
axon.Ra = 150
axon.cm = 0.5
axon.nseg = 5
axon.insert('leak')
axon.insert('NaCh')
axon.insert('HT')
#axon.insert('LT')
#axon.insert('IH')
axon.ek = soma.ek
axon.ena = soma.ena

erev = -79
gleak = 12
gklt = 50
gh = 18.8

axon.connect(soma(1))
dend.connect(soma(0))

totalcap = 25  # Total membrane capacitance in pF
somaarea = (totalcap * 1e-6) / 1  # Convert to cm^2 assuming 1 µF/cm²
axonarea = np.pi * axon.diam * axon.L * 1e-8  # in cm²
def nstomho(x):
    return (1e-9 * x / somaarea)  # Convert conductance to mho/cm²
def nstomho_axon(x):
    return (1e-9 * x / axonarea)


def extract_features(trace, time):
    dt = time[1] - time[0]
    dV = np.gradient(trace, dt)
    rest = np.mean(trace[:int(5/dt)])  # average first 5 ms
    peak_idx = np.argmax(trace)
    peak = trace[peak_idx]

    # Only consider dV/dt after 10 ms for threshold detection
    search_start_time = 11  # ms
    search_start_idx = np.searchsorted(time, search_start_time)
    dV_slice = dV[search_start_idx:]
    trace_slice = trace[search_start_idx:]
    time_slice = time[search_start_idx:]

    try:
        relative_thresh_idx = np.where(dV_slice > 45)[0][0]
        thresh_idx = search_start_idx + relative_thresh_idx
        threshold = trace[thresh_idx]
        latency = time[thresh_idx]
    except IndexError:
        threshold = np.nan
        latency = np.nan

    amp = peak - threshold

    # Half width
    half_amp = threshold + 0.5 * amp
    above_half = np.where(trace > half_amp)[0]
    if len(above_half) > 2:
        width = (above_half[-1] - above_half[0]) * dt
    else:
        width = np.nan

    # AHP (min value after peak)
    AHP = np.min(trace[peak_idx:]) if peak_idx < len(trace) else np.nan

    return {
        'rest': rest,
        'peak': peak,
        'amp': amp,
        'threshold': threshold,
        'latency': latency,
        'width': width,
        'AHP': AHP
    }

def feature_cost(sim_trace, exp_trace, time):
    sim_feat = extract_features(sim_trace, time)
    exp_feat = extract_features(exp_trace, time)
    weights = {
        'peak': 10,  # Increase penalty on overshoot
        'amp': 1,
        'width': 10,
        'threshold': 1,  # Strong push toward threshold match
        'latency':1,
        'AHP': 10
    }
    error = 0
    for k in weights:
        if not np.isnan(sim_feat[k]) and not np.isnan(exp_feat[k]):
            error += weights[k] * ((sim_feat[k] - exp_feat[k]) ** 2)
    return error

def interpolate_simulation(t_neuron, v_neuron, t_exp):
    interp_func = interp1d(t_neuron, v_neuron, kind='cubic', fill_value='extrapolate')
    return interp_func(t_exp)

def penalty_terms(v_sim):
    peak = np.max(v_sim)
    rest = v_sim[0]
    penalty = 0
    if peak < -10 or peak > 10:
        penalty += 100
    if rest > -40 or rest < -80:
        penalty += 1000
    return penalty

def run_simulation(gna, gkht, gklt, stim_amp=0.5, stim_dur=300, stim_delay=50):
    set_conductances(gna, gkht, gklt, gh, erev, gleak)

    stim = h.IClamp(soma(0.5))
    stim.delay = stim_delay
    stim.dur = stim_dur
    stim.amp = stim_amp

    h.dt = 0.02
    h.steps_per_ms = int(1.0 / h.dt)
    t_vec = h.Vector().record(h._ref_t)
    v_vec = h.Vector().record(soma(0.5)._ref_v)
    v_vec_axon = h.Vector().record(axon(0.5)._ref_v)

    h.v_init = v_init
    mFun.custom_init(v_init)
    h.tstop = stim.delay + stim_dur + 100
    h.continuerun(h.tstop)

    return np.array(t_vec), np.array(v_vec), np.array(v_vec_axon)

def set_conductances(gna, gkht, gklt, gh, erev, gleak, axon_scale=1.2):
    soma.gnabar_NaCh = nstomho(gna) * 0.01
    soma.gkhtbar_HT = nstomho(gkht) * 0.5
    soma.gkltbar_LT = nstomho(gklt)
    soma.ghbar_IH = nstomho(gh)
    soma.erev_leak = erev
    soma.g_leak = nstomho(gleak)

    axon.gnabar_NaCh = nstomho_axon(gna) * axon_scale
    axon.gkhtbar_HT = nstomho_axon(gkht) * 1.5
    axon.erev_leak = erev
    axon.g_leak = nstomho_axon(gleak)

    for seg in dend:
        seg.g_leak = nstomho(gleak)
        seg.erev_leak = erev

def sweep_cost(params, stim_amp, v_exp, t_exp):
    gna, gkht, gklt, gh, gleak = params
    t_sim, v_sim, _ = run_simulation(gna, gkht, gklt, stim_amp)
    v_interp = interpolate_simulation(t_sim, v_sim, t_exp)
    mse = np.mean((v_interp - v_exp) ** 2)
    penalty = penalty_terms(v_interp)
    return mse + penalty

def full_sweep_cost(params):
    gna, gkht, gklt, gh, gleak = params
    set_conductances(gna, gkht, gklt, gh, erev, gleak)
    total_error = 0
    for i in valid_sweeps:
        v_exp = voltage[i] * 1000  # mV
        t_exp = time[i] * 1000  # ms
        stim_amp = float(np.max(stim[i]))  # assuming in nA

        t_sim, v_sim, _ = run_simulation(gna, gkht, gklt, stim_amp)
        v_interp = interpolate_simulation(t_sim, v_sim, t_exp)
        mse = np.mean((v_interp - v_exp) ** 2)
        penalty = penalty_terms(v_interp)
        total_error += mse + penalty

    return total_error / len(valid_sweeps)

# === Optimization ===
bounds = [
    (50, 900),   # gNa
    (10, 700),   # gKHT
    (10, 300),   # gKLT
    (5, 50),     # gH
    (5, 50),     # gLeak
]


def main():
    start = timer.time()
    print("\nFitting conductances using all sweeps...\n")

    result_global = differential_evolution(
        full_sweep_cost,
        bounds,
        strategy='best1bin',
        maxiter=5,
        popsize=6,
        workers=4,  # <- multiprocessing
        updating='deferred'
    )
    result_local = minimize(
        full_sweep_cost,
        result_global.x,
        bounds=bounds,
        method='L-BFGS-B',
        options={'maxiter': 200}
    )

    opt_gna, opt_gkht, opt_gklt, opt_gh, opt_gleak = result_local.x

    print(f"Optimal gNa: {opt_gna:.2f}, gKHT: {opt_gkht:.2f}, gKLT: {opt_gklt:.2f}, gH: {opt_gh:.2f}, gLeak: {opt_gleak:.2f}")

    # Plot results
    plt.figure(figsize=(12, 6))
    for i in valid_sweeps:
        try:
            v_exp = voltage[i] * 1000
            t_exp = time[i] * 1000
            stim_amp = float(np.max(np.array(stim[i], dtype=float)))

            t_sim, v_sim, _ = run_simulation(opt_gna, opt_gkht, opt_gklt, stim_amp)
            v_interp = interpolate_simulation(t_sim, v_sim, t_exp)

            plt.plot(t_exp, v_exp, alpha=0.4)
            plt.plot(t_exp, v_interp, '--', alpha=0.6)

            # Optional: print MSE for each sweep
            mse = np.mean((v_interp - v_exp) ** 2)
            print(f"Sweep {i}: MSE = {mse:.2f}")

        except Exception as e:
            print(f"[WARN] Skipped sweep {i} due to error: {e}")

    plt.title('Fits for All Sweeps')
    plt.xlabel('Time (ms)')
    plt.ylabel('Membrane potential (mV)')
    plt.grid()
    plt.tight_layout()

    print("\nShowing plots...")
    plt.show(block=True)
    end = timer.time()
    print(f"\nTotal runtime: {end - start:.2f} seconds")

if __name__ == "__main__":
    main()
