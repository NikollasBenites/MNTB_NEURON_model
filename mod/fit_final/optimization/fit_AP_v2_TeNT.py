import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize, differential_evolution
from neuron import h
from collections import namedtuple
import MNTB_PN_myFunctions as mFun
import datetime
from MNTB_PN_fit import MNTB
from sklearn.metrics import mean_squared_error, r2_score
import time

ParamSet = namedtuple("ParamSet", [
    "gna", "gkht", "gklt", "gh", "gka", "gleak", "stim_amp","cam","kam","cbm","kbm"
])

h.load_file('stdrun.hoc')
np.random.seed(42)
script_dir = os.path.dirname(os.path.abspath(__file__))
param_file_path = os.path.join(script_dir, "..","results","_fit_results","passive_params_experimental_data_02062024_P9_FVB_PunTeTx_Dan_TeNT_80pA_S4C1_CC Test Old1_20250606_144815.txt")
filename = "sweep_11_clipped_510ms_02062024_P9_FVB_PunTeTx_Dan_TeNT_120pA_S4C1.csv"

if not os.path.exists(param_file_path):
    raise FileNotFoundError(f"Passive parameters not found at: {param_file_path}")
with open(param_file_path, "r") as f:
    gleak, gklt, gh, erev, gkht, gna, gka = map(float, f.read().strip().split(","))


# === Create Output Folder ===
file = filename.split(".")[0]
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = os.path.join(os.getcwd(),"..", "results", f"fit_AP_{file}_{timestamp}")
os.makedirs(output_dir, exist_ok=True)

# Ask for expected phenotype at +50 pA
try:
    expected_pattern = input("At +50 pA above rheobase, is the neuron phasic or tonic? ").strip().lower()
except EOFError:
    expected_pattern = "phasic"
assert expected_pattern in ["phasic", "tonic"], "Please enter 'phasic' or 'tonic'."

# Load experimental data
data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", filename))
experimentalTrace = np.genfromtxt(data_path, delimiter=',', skip_header=1, dtype=float, filling_values=np.nan)

t_exp = experimentalTrace[:,0] # ms
t_exp = t_exp - t_exp[0]
samples = np.sum((t_exp >= 0) & (t_exp <= 0.002))
if samples > 2:
    t_exp = t_exp*1000
    print("t_exp converted to ms")

V_exp = experimentalTrace[:,1]  # mV
fp = V_exp[0]
if abs(fp) < 1:
    V_exp *= 1000
    print("V_exp converted to mV")

from scipy.signal import butter, filtfilt

def butter_lowpass(cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    if not 0 < normal_cutoff < 1:
        raise ValueError(f"⚠️ Invalid normalized cutoff: {normal_cutoff:.3f} (fs={fs}, cutoff={cutoff})")
    return butter(order, normal_cutoff, btype='low', analog=False)

def lowpass_filter(data, cutoff=2000, fs=50000, order=4):
    b, a = butter_lowpass(cutoff, fs, order=order)
    return filtfilt(b, a, data)

# === Compute sampling frequency from ms → Hz
fs = 1000 / (t_exp[1] - t_exp[0])  # Correct fs in Hz
#V_exp = lowpass_filter(V_exp, cutoff=2000, fs=fs)
#print(f"✅ Applied low-pass filter at 2 kHz (fs = {fs:.1f} Hz)")


# Define soma parameters
totalcap = 25  # Total membrane capacitance in pF for the cell (input capacitance)
somaarea = (totalcap * 1e-6) / 1  # pf -> uF,assumes 1 uF/cm2; result is in cm2

ek = -106.81
ena = 62.77

################# sodium kinetics
cam = 76.4 #76.4
kam = .037
cbm = 6.930852 #6.930852
kbm = -.043

lbkna = 0.5
hbkna = 1.5

cell = MNTB(0,somaarea,erev,gleak,ena,gna,gh,gka,gklt,gkht,ek,cam,kam,cbm,kbm)

stim_dur = 300
stim_delay = 10

stim_amp = 0.120
lbamp = 0.5
hbamp = 1.5

# gleak = gleak
lbleak = 0.5
hbleak = 1.5

gkht = 200
lbKht = 0.5
hbKht = 1.5

lbKlt = 0.1
hbKlt = 1.9

gka = 100
lbka = 0.1
hbka = 1.9

lbih = 0.5
hbih = 1.5

gna = 200
lbgNa = 0.1
hbgNa = 1.9

bounds = [
    (gna*lbgNa, gna*hbgNa),             # gNa
    (gkht * lbKht, gkht * hbKht),
    (gklt * lbKlt, gklt * hbKlt),       # gKLT
    (gh * lbih, gh * hbih),             # gIH
    (gka * lbka, gka * hbka),           # gka
    (gleak * lbleak, gleak * hbleak),   # gleak
    (stim_amp*lbamp, stim_amp*hbamp),  # stim-amp
    (cam*lbkna, cam*hbkna),
    (kam*lbkna, kam*hbkna),
    (cbm*lbkna, cbm*hbkna),
    (kbm*hbkna, kbm*lbkna)
]


def extract_features(trace, time):
    dt = time[1] - time[0]

    # Build spline for smooth derivative calculation
    spline = CubicSpline(time, trace, bc_type='natural', extrapolate=True)
    voltage = spline(time)
    dvdt = spline(time, nu=1)

    # === Resting potential (first 5 ms)
    rest = np.mean(voltage[:int(5 / dt)])

    # === Peak
    peak_idx = np.argmax(voltage)
    peak = voltage[peak_idx]

    # === Threshold detection: where dV/dt > 25 after 11 ms
    start_idx = int(11 / dt)
    try:
        rel_thresh_idx = np.where(dvdt[start_idx:] > 25)[0][0]
        thresh_idx = start_idx + rel_thresh_idx
        threshold = voltage[thresh_idx]
        latency = time[thresh_idx]
    except IndexError:
        return {
            'rest': rest, 'peak': peak, 'amp': np.nan,
            'threshold': np.nan, 'latency': np.nan,
            'width': np.nan, 'AHP': np.nan
        }

    amp = peak - threshold
    half_amp = threshold + 0.5 * amp

    # === Width at half-amplitude
    above_half = np.where(
        (voltage > half_amp) &
        (np.arange(len(voltage)) > thresh_idx) &
        (np.arange(len(voltage)) < peak_idx + int(5 / dt))
    )[0]

    width = (above_half[-1] - above_half[0]) * dt if len(above_half) > 1 else np.nan

    # === After-hyperpolarization
    AHP = np.min(voltage[peak_idx:]) if peak_idx < len(voltage) else np.nan

    return {
        'rest': rest,
        'peak': peak,
        'amp': amp,
        'threshold': threshold,
        'latency': latency,
        'width': width,
        'AHP': AHP
    }

def feature_cost(sim_trace, exp_trace, time, return_details=False):
    sim_feat = extract_features(sim_trace, time)
    exp_feat = extract_features(exp_trace, time)
    weights = {
        'rest':      1.0,
        'peak':      1.0,
        'amp':       1.0,
        'threshold': 1.0,
        'latency':   1.0,
        'width':     1.0,
        'AHP':       1.0
    }

    error = 0
    details = {}
    for k in weights:
        if not np.isnan(sim_feat[k]) and not np.isnan(exp_feat[k]):
            diff_sq = (sim_feat[k] - exp_feat[k])**2
            weighted = weights[k] * diff_sq
            error += weighted
            details[k] = {
                'sim': sim_feat[k],
                'exp': exp_feat[k],
                'diff²': diff_sq,
                'weighted_error': weighted
            }
        else:
            details[k] = {
                'sim': sim_feat.get(k, np.nan),
                'exp': exp_feat.get(k, np.nan),
                'diff²': np.nan,
                'weighted_error': np.nan
            }

    if return_details:
        return error, details
    else:
        return error


#@lru_cache(maxsize=None)
def run_simulation(p: ParamSet, stim_dur=300, stim_delay=10):
    """
    Unified simulation wrapper for fitting, using run_unified_simulation.
    """
    v_init = -70
    totalcap = 25  # pF
    somaarea = (totalcap * 1e-6) / 1  # cm² assuming 1 µF/cm²
    h.celsius = 35
    param_dict = {
        "gna": p.gna,
        "gkht": p.gkht,
        "gklt": p.gklt,
        "gh": p.gh,
        "gka": p.gka,
        "gleak": p.gleak,
        "cam": p.cam,
        "kam": p.kam,
        "cbm": p.cbm,
        "kbm": p.kbm,
        "erev": erev,
        "ena": ena,
        "ek": ek,
        "somaarea": somaarea
    }

    t, v = mFun.run_unified_simulation(
        MNTB_class=MNTB,
        param_dict=param_dict,
        stim_amp=p.stim_amp,
        stim_delay=stim_delay,
        stim_dur=stim_dur,
        v_init=v_init,
        total_duration=stim_delay + stim_dur + 200,
        return_stim=False
    )

    return t, v

# def interpolate_simulation(t_neuron, v_neuron, t_exp):
#     interp_func = interp1d(t_neuron, v_neuron, kind='cubic', fill_value='extrapolate')
#     return interp_func(t_exp)

def interpolate_simulation(t_neuron, v_neuron, t_exp):
    # Ensure numpy arrays
    t_neuron = np.asarray(t_neuron)
    v_neuron = np.asarray(v_neuron)
    t_exp = np.asarray(t_exp)

    # Sort the time and voltage arrays (required for spline)
    sort_idx = np.argsort(t_neuron)
    t_neuron = t_neuron[sort_idx]
    v_neuron = v_neuron[sort_idx]

    # Create cubic spline with extrapolation enabled
    spline = CubicSpline(t_neuron, v_neuron, bc_type='natural', extrapolate=True)

    # Evaluate spline at experimental time points (t_exp)
    v_interp = spline(t_exp)

    return v_interp


def penalty_terms(v_sim):
    peak = np.max(v_sim)
    rest = v_sim[0]
    penalty = 0
    if peak < -15 or peak > 20:
        penalty += 1
    if rest > -55 or rest < -90:
        penalty += 1000
    return penalty

def cost_function(params): #no ap window
    assert len(params) == 11, "Mismatch in number of parameters"
    p = ParamSet(*params)

    t_sim, v_sim = run_simulation(p)

    v_interp = interpolate_simulation(t_sim, v_sim, t_exp)
    exp_feat = extract_features(V_exp, t_exp)
    sim_feat = extract_features(v_interp, t_exp)
    # Time shift between peaks
    dt = t_exp[1] - t_exp[0]
    time_shift = abs(np.argmax(v_interp) - np.argmax(V_exp)) * dt
    weight = 50  # you can tune this weight
    time_error = weight * time_shift

    mse = np.mean((v_interp - V_exp)**2)
    f_cost = feature_cost(v_interp, V_exp, t_exp)
    penalty = penalty_terms(v_interp)
    peak_penalty = 0
    # sim_peak = np.max(v_interp)
    # if sim_peak > 5:
    #     peak_penalty += 10 * (sim_peak - 20)**2

    alpha = 1  # weight for MSE
    beta =  1 # weight for feature cost

    total_cost = alpha * mse + beta * f_cost + time_error + penalty + peak_penalty

    return total_cost

def cost_function1(params):
    assert len(params) == 11, "Mismatch in number of parameters"
    p = ParamSet(*params)

    # Run simulation
    t_sim, v_sim = run_simulation(p)

    # Interpolate to match experimental time resolution
    v_interp = interpolate_simulation(t_sim, v_sim, t_exp)

    # === Extract AP region ===
    exp_feat = extract_features(V_exp, t_exp)
    sim_feat = extract_features(v_interp, t_exp)

    # If no AP detected in either, return a large penalty
    if np.isnan(exp_feat['latency']) or np.isnan(sim_feat['latency']):
        return 1e6

    # Define AP window (2 ms before threshold to 30 ms after peak)
    dt = t_exp[1] - t_exp[0]
    try:
        ap_start = max(0, int((exp_feat['latency'] - 3) / dt))
        ap_end = min(len(t_exp), int((exp_feat['latency'] + 20) / dt))
    except Exception:
        return 1e6

    v_interp_ap = v_interp[ap_start:ap_end]
    v_exp_ap = V_exp[ap_start:ap_end]
    t_ap = t_exp[ap_start:ap_end]

    # Feature cost only within AP window
    f_cost = feature_cost(v_interp_ap, v_exp_ap, t_ap)

    # MSE only in AP window
    mse = np.mean((v_interp_ap - v_exp_ap) ** 2)

    # Time shift of peaks (only within window)
    time_shift = abs(np.argmax(v_interp_ap) - np.argmax(v_exp_ap)) * dt
    time_error = 500 * time_shift

    # Resting potential penalty still on full trace
    penalty = penalty_terms(v_interp)

    # Total weighted cost
    alpha = 1     # MSE
    beta =  1     # Feature cost

    total_cost = alpha * mse + beta * f_cost + time_error + penalty

    return total_cost

def log_and_plot_optimization(result_global, result_local, param_names=None, save_path=None):
    """
    Compare and plot cost function results from global and local optimizations.

    Parameters:
    - result_global: output from differential_evolution
    - result_local: output from minimize
    - param_names: list of parameter names (optional)
    - save_path: if provided, save log CSV and plot there
    """
    global_cost = result_global.fun
    local_cost = result_local.fun
    delta = global_cost - local_cost

    print(f"🔍 Global Cost: {global_cost:.6f}")
    print(f"🔧 Local Cost:  {local_cost:.6f}")
    print(f"📉 Improvement: {delta:.6f}")

    # Combine parameter sets
    if param_names is None:
        param_names = [f'p{i}' for i in range(len(result_local.x))]

    df = pd.DataFrame({
        'Parameter': param_names,
        'Global_fit': result_global.x,
        'Local_fit': result_local.x
    })
    df['Change'] = df['Local_fit'] - df['Global_fit']

    print("\nParameter changes:\n", df)

    # Optional save
    if save_path:
        csv_path = f"{save_path}/fit_log.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n✅ Saved log to {csv_path}")

    # Optional bar plot of change
    plt.figure(figsize=(8, 4))
    plt.bar(df['Parameter'], df['Change'], color='skyblue')
    plt.axhline(0, linestyle='--', color='gray')
    plt.ylabel("Change (Local - Global)")
    plt.title("Parameter Adjustment After Local Fit")
    plt.tight_layout()

    if save_path:
        fig_path = f"{save_path}/fit_comparison_plot.png"
        plt.savefig(fig_path, dpi=300)
        print(f"📊 Saved plot to {fig_path}")
    else:
        plt.show()

def create_local_bounds(center, rel_window=0.1, abs_min=None, abs_max=None):
    """Create a tuple (min, max) around center using ±rel_window, ensuring valid order even for negative center."""
    lower = center * (1 - rel_window)
    upper = center * (1 + rel_window)

    # Ensure correct order
    bound_min, bound_max = min(lower, upper), max(lower, upper)

    # Clamp to absolute bounds if given
    if abs_min is not None:
        bound_min = max(bound_min, abs_min)
    if abs_max is not None:
        bound_max = min(bound_max, abs_max)

    return (bound_min, bound_max)


print("Running optimization...")
t0 = time.time()
result_global = differential_evolution(cost_function, bounds, strategy='best1bin', maxiter=5, popsize=50, mutation=1.0, updating='deferred',polish=False, tol=1e-2)
t1 = time.time()
print(f"✅ Global optimization done in {t1 - t0:.2f} seconds")
print("Running minimization...")
t2 = time.time()
result_local = minimize(cost_function, result_global.x, bounds=bounds, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-6, 'disp': True})
t3 = time.time()
print(f"✅ Local minimization done in {t3 - t2:.2f} seconds")
print(f"🕒 Total optimization time: {t3 - t0:.2f} seconds")

def run_refinement_loop(initial_result, cost_func, rel_windows, max_iters=150, min_delta=1e-6):
    history = [initial_result.fun]
    current_result = initial_result

    print("\n🔁 Starting refinement loop:")
    for i in range(max_iters):
        print(f"\n🔂 Iteration {i+1}")

        x_opt = current_result.x
        new_bounds = [
            create_local_bounds(x_opt[j], rel_window=rel_windows[j])
            for j in range(len(x_opt))
        ]

        new_result = minimize(
            cost_func,
            x_opt,
            method='L-BFGS-B',
            bounds=new_bounds,
            options={'maxiter': 2000, 'ftol': 1e-6, 'disp': False}
        )

        delta = current_result.fun - new_result.fun
        history.append(new_result.fun)

        print(f"   Cost: {current_result.fun:.4f} → {new_result.fun:.6f} (Δ = {delta:.6f})")

        if delta < min_delta:
            print("   ✅ Converged: small improvement.")
            break

        current_result = new_result

    return current_result, history


def count_spikes(trace, time, threshold=-15):
    """
    Count number of spikes based on upward threshold crossings.
    """
    above = trace > threshold
    crossings = np.where(np.diff(above.astype(int)) == 3)[0]
    return len(crossings)

def check_and_refit_if_needed(params_opt, expected_pattern, t_exp, V_exp, rel_windows, output_dir):
    def simulate_plus_50(p):
        stim_amp_plus_50 = p.stim_amp + 0.050
        test_p = p._replace(stim_amp=stim_amp_plus_50)
        t_hi, v_hi = run_simulation(test_p)
        n_spikes = count_spikes(v_hi, t_hi)
        pattern = "phasic" if n_spikes == 1 else "tonic"
        return pattern, n_spikes, t_hi, v_hi

    # --- Initial test
    observed_pattern, n_spikes, t_hi, v_hi = simulate_plus_50(params_opt)

    print(f"\n🔍 Verifying +50 pA response:")
    print(f"Expected: {expected_pattern}, Observed: {observed_pattern} ({n_spikes} spike{'s' if n_spikes != 1 else ''})")

    if observed_pattern == expected_pattern:
        print("✅ Match confirmed. No re-optimization needed.")
        return params_opt, False, t_hi, v_hi, observed_pattern

    print("❌ Mismatch detected. Re-optimizing selected channels...")

    # === Only optimize: gna, gkht, gklt, gka
    # Full dict of fixed parameters
    fixed_dict = params_opt._asdict().copy()

    # Extract the subset to optimize
    param_names = ['gna', 'gkht', 'gklt', 'gka']
    x0 = [fixed_dict[k] for k in param_names]

    # Remove keys we're going to refit
    fixed = {k: v for k, v in fixed_dict.items() if k not in param_names}

    broader_bounds = [
        (fixed_dict['gna'] * 0.5, fixed_dict['gna'] * 1.3),
        (fixed_dict['gkht'] * 0.3, fixed_dict['gkht'] * 1.7),
        (fixed_dict['gklt'] * 0.5, fixed_dict['gklt'] * 1.5),
        (fixed_dict['gka'] * 0.5, fixed_dict['gka'] * 1.5)
    ]

    def cost_partial(x):
        pdict = fixed.copy()
        pdict.update(dict(zip(param_names, x)))
        return cost_function(ParamSet(**pdict))

    result_global = differential_evolution(cost_partial, broader_bounds, strategy='best1bin', maxiter=5, popsize=50, mutation=1.0, updating='deferred',polish=False, tol=1e-2)
    result_local = minimize(cost_partial, result_global.x, bounds=broader_bounds, method='L-BFGS-B', options={'maxiter': 1000,'ftol': 1e-6 ,'disp': True})

    # Build new full ParamSet
    # Merge back the optimized params
    updated = fixed.copy()
    updated.update(dict(zip(param_names, result_local.x)))

    # Now reconstruct full ParamSet
    new_params = ParamSet(**updated)

    # Final re-test
    final_pattern, final_spikes, t_hi, v_hi = simulate_plus_50(new_params)
    print(f"\n✅ Final re-test at +50 pA: {final_spikes} spike(s) — {final_pattern.upper()} firing")

    if final_pattern != expected_pattern:
        print("⚠️  WARNING: Still mismatch after refitting.")
    else:
        print("🎯 Final model now matches expected pattern.")

    # Save summary
    summary_path = os.path.join(output_dir, "refit_summary.json")
    summary = {
        "reoptimized": True,
        "expected_pattern": expected_pattern,
        "observed_before": observed_pattern,
        "observed_after": final_pattern,
        "n_spikes_after": final_spikes,
        "stim_amp_plus_50": new_params.stim_amp + 0.050
    }
    with open(summary_path, "w") as f:
        import json
        json.dump(summary, f, indent=4)
    print(f"📝 Saved refit summary to {summary_path}")

    return new_params, True, t_hi, v_hi, final_pattern


rel_windows = [
    0.5,  # gNa: sodium conductance — narrow ±10%
    0.3,  # gKHT: high-threshold K⁺ conductance — broader ±50%
    0.5,  # gKLT: low-threshold K⁺ conductance — broader ±50%
    0.5,  # gIH: HCN conductance — narrow ±10%
    0.5,  # gKA: A-type K⁺ conductance — narrow ±10%
    0.1,  # gLeak: leak conductance — narrow ±10%
    0.1,  # stim_amp: current amplitude — broader ±50%
    0.1,  # cam: Na⁺ activation slope — narrow ±10%
    0.1,  # kam: Na⁺ activation V-half — narrow ±10%
    0.1,  # cbm: Na⁺ inactivation slope — narrow ±10%
    0.1   # kbm: Na⁺ inactivation V-half — narrow ±10%
]


result_local_refined, cost_history = run_refinement_loop(result_local, cost_function, rel_windows)

params_opt = ParamSet(*result_local_refined.x)
print("Optimized parameters:")
for name, value in params_opt._asdict().items():
    print(f"{name}: {value:.4f}")

params_opt, reoptimized, t_hi, v_hi, final_pattern = check_and_refit_if_needed(
    params_opt, expected_pattern, t_exp, V_exp, rel_windows, output_dir
)

param_names = ParamSet._fields

log_and_plot_optimization(result_global, result_local, param_names, save_path="/Users/nikollas/Library/CloudStorage/OneDrive-UniversityofSouthFlorida/MNTB_neuron/mod/fit_final/results/_fit_results")
log_and_plot_optimization(result_local, result_local_refined, param_names,save_path="/Users/nikollas/Library/CloudStorage/OneDrive-UniversityofSouthFlorida/MNTB_neuron/mod/fit_final/results/_fit_results")
#
print(f"Best stim-amp: {params_opt.stim_amp:.2f} pA")
print(f" Optimized gna: {params_opt.gna:.2f}, gklt: {params_opt.gklt: .2f}, gkht: {params_opt.gkht: .2f}), gh: {params_opt.gh:.2f}, gka:{params_opt.gka:.2f}, gleak: {params_opt.gleak:.2f}, "
      f"cam: {params_opt.cam: .2f}, kam: {params_opt.kam: .2f}, cbm: {params_opt.cbm: .2f}, kbm: {params_opt.kbm: .2f}")

# Final simulation and plot
t_sim, v_sim = run_simulation(params_opt)

# Interpolate simulated trace to match experimental time points
v_interp = interpolate_simulation(t_sim, v_sim, t_exp)

# Compute fit quality metrics
mse = mean_squared_error(V_exp, v_interp)
r2 = r2_score(V_exp, v_interp)
time_shift = abs(np.argmax(v_interp) - np.argmax(V_exp)) * (t_exp[1] - t_exp[0])
feature_error = feature_cost(v_interp, V_exp, t_exp)

# Add fit quality label
fit_quality = 'good' if r2 > 0.9 and time_shift < 0.5 else 'poor'

feat_sim = extract_features(v_sim, t_sim)
print("Simulate Features:")
for k, v in feat_sim.items():
    print(f"{k}: {v:.2f}")

feat_exp = extract_features(V_exp, t_exp)
print("Experimental Features:")
for k, v in feat_exp.items():
    print(f"{k}: {v:.2f}")

results = params_opt._asdict()
results.update(feat_sim)
results['mse'] = mse
results['r2'] = r2
results['time_shift'] = time_shift
results['feature_error'] = feature_error
results['fit_quality'] = fit_quality
print(f"\n=== Fit Quality Metrics ===")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R-squared (R²):           {r2:.4f}")
print(f"Time Shift (ms):          {time_shift:.4f}")
print(f"Feature Error:            {feature_error:.2f}")
print(f"Fit Quality:              {fit_quality}")

f_error, f_details = feature_cost(v_interp, V_exp, t_exp, return_details=True)

print("\n📊 Feature-wise Error Breakdown:")
for feat, vals in f_details.items():
    print(f"{feat:10s} | Sim: {vals['sim']:.2f} | Exp: {vals['exp']:.2f} | Diff²: {vals['diff²']:.2f} | Weighted: {vals['weighted_error']:.2f}")

pd.DataFrame([results]).to_csv(os.path.join(output_dir,f"fit_results_{timestamp}.csv"), index=False)

results_exp = {k: v for k, v in results.items() if k in feat_exp}

df = pd.DataFrame([results_exp])  # Create DataFrame first
df = pd.DataFrame([results_exp]).to_csv(os.path.join(output_dir,f"fit_results_exp_{timestamp}.csv"), index=False)

combined_results = {
    "gleak": gleak, "gklt": params_opt.gklt, "gh": params_opt.gh, "erev": erev,
    "gna": params_opt.gna, "gkht": params_opt.gkht, "gka": params_opt.gka,
    "stim_amp": params_opt.stim_amp,
    "cam": params_opt.cam, "kam": params_opt.kam, "cbm": params_opt.cbm, "kbm": params_opt.kbm
}

pd.DataFrame([combined_results]).to_csv(os.path.join(script_dir, "..","results","_fit_results", f"all_fitted_params_{file}_{timestamp}.csv"), index=False)
pd.DataFrame([combined_results]).to_csv(os.path.join(script_dir, "..","results","_fit_results", f"all_fitted_params.csv"), index=False) #the last
# monitor_cache_size()
plt.figure(figsize=(10, 5))
plt.plot(t_exp, V_exp, label='Experimental', linewidth=2)
plt.plot(t_sim, v_sim, label='Simulated (fit)', linestyle='--')
plt.legend()
plt.xlabel('Time (ms)')
plt.ylabel('Membrane potential (mV)')
plt.title('Action Potential Fit')
thresh_exp = extract_features(V_exp, t_exp)['latency']
thresh_sim = extract_features(v_sim, t_sim)['latency']
plt.axvline(thresh_exp, color='blue', linestyle=':', label='Exp Threshold')
plt.axvline(thresh_sim, color='orange', linestyle=':', label='Sim Threshold')
plt.tight_layout()
plt.show()

half_amp = feat_sim['threshold'] + 0.5 * feat_sim['amp']
plt.plot(t_sim, v_sim)
plt.axhline(half_amp, color='red', linestyle='--', label='Half amplitude')
plt.title("Check AP width region")
plt.legend()
plt.show()

# Save +50 pA trace
trace_df = pd.DataFrame({
    "time_ms": t_hi,
    "voltage_mV": v_hi
})
trace_file = os.path.join(output_dir, f"sim_trace_plus_50pA_{final_pattern}.csv")
trace_df.to_csv(trace_file, index=False)
print(f"💾 Saved +50 pA trace to {trace_file}")

# === Plot clipped AP window ===
dt = t_exp[1] - t_exp[0]
try:
    ap_start = max(0, int((feat_exp['latency'] - 3) / dt))
    ap_end = min(len(t_exp), int((feat_exp['latency'] + 50) / dt))
except Exception as e:
    print("⚠️ Could not clip AP window:", e)
    ap_start = 0
    ap_end = len(t_exp)

t_clip = t_exp[ap_start:ap_end]
v_exp_clip = V_exp[ap_start:ap_end]
v_sim_clip = interpolate_simulation(t_sim, v_sim, t_clip)

plt.figure(figsize=(8, 4))
plt.plot(t_clip, v_exp_clip, label='Experimental (filtered)', linewidth=2)
plt.plot(t_clip, v_sim_clip, '--', label='Simulated (fit)', linewidth=2)
plt.xlabel("Time (ms)")
plt.ylabel("Membrane potential (mV)")
plt.title("AP Fit — Clipped to AP Window")
plt.legend()
plt.tight_layout()
plt.show()

# === Align by threshold (time and voltage) ===
thresh_exp_time = feat_exp['latency']
thresh_sim_time = feat_sim['latency']
thresh_exp_voltage = feat_exp['threshold']
thresh_sim_voltage = feat_sim['threshold']

# Time alignment: shift time vectors so threshold is at 0
t_clip_aligned_exp = t_clip - thresh_exp_time
t_clip_aligned_sim = t_clip - thresh_sim_time

# Voltage alignment: subtract threshold voltage so both traces start at 0 mV
v_exp_aligned = v_exp_clip - thresh_exp_voltage
v_sim_aligned = v_sim_clip - thresh_sim_voltage

# === Plot aligned APs
plt.figure(figsize=(8, 4))
plt.plot(t_clip_aligned_exp, v_exp_aligned, label='Experimental (aligned)', linewidth=2)
plt.plot(t_clip_aligned_sim, v_sim_aligned, '--', label='Simulated (aligned)', linewidth=2)
plt.axvline(0, color='gray', linestyle=':', label='Threshold time')
plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
plt.xlabel("Time (ms, aligned to threshold)")
plt.ylabel("Membrane potential (mV, threshold-normalized)")
plt.title("AP Fit — Aligned by Threshold Time and Voltage")
plt.legend()
plt.tight_layout()
# Save to PDF
aligned_pdf_path = os.path.join(output_dir, f"aligned_AP_fit_{filename}_{timestamp}.pdf")
plt.savefig(aligned_pdf_path, format='pdf')
print(f"📄 Saved aligned AP plot to: {aligned_pdf_path}")
plt.show()

