import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# === Voltage and temperature setup ===
V = np.linspace(-100, 50, 300)
celsius = 35
q10 = 3 ** ((celsius - 22) / 10)

# === Define your gating functions from ka.mod ===
def a_inf(v): return (1 / (1 + np.exp(-(v + 40) / 6)))**0.25
def a_tau(v): return (500 / (7 * np.exp((v + 60) / 14) + 29 * np.exp(-(v + 60) / 24))) + 0.1
def b_inf(v): return (1 / (1 + np.exp((v + 66) / 7)))**0.5
def b_tau(v): return (1000 / (14 * np.exp((v + 60) / 27) + 29 * np.exp(-(v + 60) / 24))) + 1
def c_tau(v): return (90 / (1 + np.exp((-66 - v) / 17))) + 10

# === Apply Q10 correction ===
atau = a_tau(V) / q10
btau = b_tau(V) / q10
ctau = c_tau(V) / q10

# === Compute steady-state values and alpha/beta for a, b, c
ainf = a_inf(V)
binf = b_inf(V)
cinf = binf

alpha_a, beta_a = ainf / atau, (1 - ainf) / atau
alpha_b, beta_b = binf / btau, (1 - binf) / btau
alpha_c, beta_c = cinf / ctau, (1 - cinf) / ctau

# === Define exponential fitting function
def exp_fit(v, c, k): return c * np.exp(k * v)

# === Fit exponential functions
popt_alpha_a, _ = curve_fit(exp_fit, V, alpha_a, p0=(0.1, 0.01), maxfev=10000)
popt_beta_a, _ = curve_fit(exp_fit, V, beta_a, p0=(0.1, -0.01), maxfev=10000)
popt_alpha_b, _ = curve_fit(exp_fit, V, alpha_b, p0=(0.1, 0.01), maxfev=10000)
popt_beta_b, _ = curve_fit(exp_fit, V, beta_b, p0=(0.1, -0.01), maxfev=10000)
popt_alpha_c, _ = curve_fit(exp_fit, V, alpha_c, p0=(0.1, 0.01), maxfev=10000)
popt_beta_c, _ = curve_fit(exp_fit, V, beta_c, p0=(0.1, -0.01), maxfev=10000)

# === Define Boltzmann function for fitting
def boltzmann_fit(v, v_half, k, min_val, max_val):
    return min_val + (max_val - min_val) / (1 + np.exp((v_half - v) / k))

# === Fit alpha and beta rates with Boltzmann functions (only meaningful for sigmoidal data)
from scipy.optimize import curve_fit

# We apply the fit to a subset that looks sigmoidal, like ainf or binf, not alpha/beta
def try_fit_boltzmann(rate_vals, V):
    # Initial guess: v_half = -30, k = 10, min = min(rate), max = max(rate)
    p0 = (-30, 10, np.min(rate_vals), np.max(rate_vals))
    bounds = ([-100, 0.1, 0, 0], [50, 100, 1, 10])
    try:
        popt, _ = curve_fit(boltzmann_fit, V, rate_vals, p0=p0, bounds=bounds, maxfev=10000)
        return popt
    except RuntimeError:
        return None

# Try fitting Boltzmann to ainf, binf, cinf
fit_ainf = try_fit_boltzmann(ainf, V)
fit_binf = try_fit_boltzmann(binf, V)
fit_cinf = try_fit_boltzmann(cinf, V)


# === Print the parameters for all gates
alpha_beta_params = {
    'alpha_a': popt_alpha_a,
    'beta_a': popt_beta_a,
    'alpha_b': popt_alpha_b,
    'beta_b': popt_beta_b,
    'alpha_c': popt_alpha_c,
    'beta_c': popt_beta_c
}

# Create a DataFrame for better visualization
import pandas as pd
df_params = pd.DataFrame({
    'Parameter': alpha_beta_params.keys(),
    'c': [params[0] for params in alpha_beta_params.values()],
    'k': [params[1] for params in alpha_beta_params.values()]
})


# === Plotting
plt.figure(figsize=(12, 8))

# Activation gate a
plt.subplot(3, 2, 1)
plt.plot(V, alpha_a, label='alpha_a', color='blue')
plt.plot(V, exp_fit(V, *popt_alpha_a), '--', label='fit', color='cyan')
plt.title('Alpha - Gate a')
plt.xlabel('Voltage (mV)')
plt.ylabel('Rate (/ms)')
plt.legend()

plt.subplot(3, 2, 2)
plt.plot(V, beta_a, label='beta_a', color='red')
plt.plot(V, exp_fit(V, *popt_beta_a), '--', label='fit', color='orange')
plt.title('Beta - Gate a')
plt.xlabel('Voltage (mV)')
plt.ylabel('Rate (/ms)')
plt.legend()

# Inactivation gate b
plt.subplot(3, 2, 3)
plt.plot(V, alpha_b, label='alpha_b', color='blue')
plt.plot(V, exp_fit(V, *popt_alpha_b), '--', label='fit', color='cyan')
plt.title('Alpha - Gate b')
plt.xlabel('Voltage (mV)')
plt.ylabel('Rate (/ms)')
plt.legend()

plt.subplot(3, 2, 4)
plt.plot(V, beta_b, label='beta_b', color='red')
plt.plot(V, exp_fit(V, *popt_beta_b), '--', label='fit', color='orange')
plt.title('Beta - Gate b')
plt.xlabel('Voltage (mV)')
plt.ylabel('Rate (/ms)')
plt.legend()

# Inactivation gate c
plt.subplot(3, 2, 5)
plt.plot(V, alpha_c, label='alpha_c', color='blue')
plt.plot(V, exp_fit(V, *popt_alpha_c), '--', label='fit', color='cyan')
plt.title('Alpha - Gate c')
plt.xlabel('Voltage (mV)')
plt.ylabel('Rate (/ms)')
plt.legend()

plt.subplot(3, 2, 6)
plt.plot(V, beta_c, label='beta_c', color='red')
plt.plot(V, exp_fit(V, *popt_beta_c), '--', label='fit', color='orange')
plt.title('Beta - Gate c')
plt.xlabel('Voltage (mV)')
plt.ylabel('Rate (/ms)')
plt.legend()

plt.tight_layout()
plt.show()

# Plotting fits
plt.figure(figsize=(10, 4))

# ainf
plt.subplot(1, 3, 1)
plt.plot(V, ainf, label='ainf', color='blue')
if fit_ainf is not None:
    plt.plot(V, boltzmann_fit(V, *fit_ainf), '--', label='Boltzmann fit', color='cyan')
plt.title('ainf vs. V')
plt.xlabel('Voltage (mV)')
plt.ylabel('Steady-state')
plt.legend()

# binf
plt.subplot(1, 3, 2)
plt.plot(V, binf, label='binf', color='green')
if fit_binf is not None:
    plt.plot(V, boltzmann_fit(V, *fit_binf), '--', label='Boltzmann fit', color='lime')
plt.title('binf vs. V')
plt.xlabel('Voltage (mV)')
plt.ylabel('Steady-state')
plt.legend()

# cinf
plt.subplot(1, 3, 3)
plt.plot(V, cinf, label='cinf', color='red')
if fit_cinf is not None:
    plt.plot(V, boltzmann_fit(V, *fit_cinf), '--', label='Boltzmann fit', color='orange')
plt.title('cinf vs. V')
plt.xlabel('Voltage (mV)')
plt.ylabel('Steady-state')
plt.legend()

plt.tight_layout()
plt.show()

# === Plot steady-state values (ainf, binf, cinf) with markers at V1/2 ===
# Extract V1/2 values from Boltzmann fits
v_half_ainf = fit_ainf[0]
v_half_binf = fit_binf[0]
v_half_cinf = fit_cinf[0]

# Compute ainf, binf, cinf at respective V1/2 values
ainf_vhalf = a_inf(v_half_ainf)
binf_vhalf = b_inf(v_half_binf)
cinf_vhalf = b_inf(v_half_cinf)  # same as binf

# Compute time constants at the same voltages
atau_vhalf = a_tau(v_half_ainf) / q10
btau_vhalf = b_tau(v_half_binf) / q10
ctau_vhalf = c_tau(v_half_cinf) / q10

# Package the results
results = {
    "V1/2_ainf (mV)": v_half_ainf,
    "ainf @ V1/2": ainf_vhalf,
    "atau @ V1/2 (ms)": atau_vhalf,
    "V1/2_binf (mV)": v_half_binf,
    "binf @ V1/2": binf_vhalf,
    "btau @ V1/2 (ms)": btau_vhalf,
    "V1/2_cinf (mV)": v_half_cinf,
    "cinf @ V1/2": cinf_vhalf,
    "ctau @ V1/2 (ms)": ctau_vhalf
}

results


plt.figure(figsize=(12, 4))

# ainf
plt.subplot(1, 3, 1)
plt.plot(V, ainf, label='ainf', color='blue')
plt.axvline(v_half_ainf, color='gray', linestyle='--', label=f'V1/2 = {v_half_ainf:.2f} mV')
plt.plot(v_half_ainf, ainf_vhalf, 'o', color='black', label=f'ainf = {ainf_vhalf:.2f}')
plt.title('ainf vs Voltage')
plt.xlabel('Voltage (mV)')
plt.ylabel('Steady-State Value')
plt.legend()

# binf
plt.subplot(1, 3, 2)
plt.plot(V, binf, label='binf', color='green')
plt.axvline(v_half_binf, color='gray', linestyle='--', label=f'V1/2 = {v_half_binf:.2f} mV')
plt.plot(v_half_binf, binf_vhalf, 'o', color='black', label=f'binf = {binf_vhalf:.2f}')
plt.title('binf vs Voltage')
plt.xlabel('Voltage (mV)')
plt.legend()

# cinf
plt.subplot(1, 3, 3)
plt.plot(V, cinf, label='cinf', color='red')
plt.axvline(v_half_cinf, color='gray', linestyle='--', label=f'V1/2 = {v_half_cinf:.2f} mV')
plt.plot(v_half_cinf, cinf_vhalf, 'o', color='black', label=f'cinf = {cinf_vhalf:.2f}')
plt.title('cinf vs Voltage')
plt.xlabel('Voltage (mV)')
plt.legend()

plt.tight_layout()
plt.show()

# === Plot time constants (atau, btau, ctau) with markers at V1/2 ===
plt.figure(figsize=(12, 4))

# atau
plt.subplot(1, 3, 1)
plt.plot(V, atau, label='atau', color='blue')
plt.axvline(v_half_ainf, color='gray', linestyle='--', label=f'V1/2 = {v_half_ainf:.2f} mV')
plt.plot(v_half_ainf, atau_vhalf, 'o', color='black', label=f'atau = {atau_vhalf:.2f} ms')
plt.title('atau vs Voltage')
plt.xlabel('Voltage (mV)')
plt.ylabel('Tau (ms)')
plt.legend()

# btau
plt.subplot(1, 3, 2)
plt.plot(V, btau, label='btau', color='green')
plt.axvline(v_half_binf, color='gray', linestyle='--', label=f'V1/2 = {v_half_binf:.2f} mV')
plt.plot(v_half_binf, btau_vhalf, 'o', color='black', label=f'btau = {btau_vhalf:.2f} ms')
plt.title('btau vs Voltage')
plt.xlabel('Voltage (mV)')
plt.ylabel('Tau (ms)')
plt.legend()

# ctau
plt.subplot(1, 3, 3)
plt.plot(V, ctau, label='ctau', color='red')
plt.axvline(v_half_cinf, color='gray', linestyle='--', label=f'V1/2 = {v_half_cinf:.2f} mV')
plt.plot(v_half_cinf, ctau_vhalf, 'o', color='black', label=f'ctau = {ctau_vhalf:.2f} ms')
plt.title('ctau vs Voltage')
plt.xlabel('Voltage (mV)')
plt.ylabel('Tau (ms)')
plt.legend()

plt.tight_layout()
plt.show()
# Voltage range
V = np.linspace(-100, 50, 500)

# Inner sigmoid (activation gate without power)
sigmoid = 1 / (1 + np.exp(-(V + 40) / 6))
# ainf is the 4th root of the full activation
ainf = sigmoid ** 0.25
# Full activation curve (what contributes to conductance)
a4 = ainf ** 4  # equivalent to sigmoid

# Plotting
plt.figure(figsize=(8, 5))
plt.plot(V, sigmoid, label='Sigmoid (no power)', linestyle='--')
plt.plot(V, ainf, label='ainf = sigmoid$^{0.25}$')
plt.plot(V, a4, label='ainf$^4$ = sigmoid', linestyle=':')
plt.title('Activation Gate Function a_inf and its Relation to Sigmoid')
plt.xlabel('Voltage (mV)')
plt.ylabel('Probability')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()