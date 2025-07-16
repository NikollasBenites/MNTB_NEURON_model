import numpy as np
import matplotlib.pyplot as plt

# === Define voltage range
v = np.linspace(-100, 0, 1000)

# === Simulation temperature
celsius = 35

# === Q10 settings
q10_kv1 = 3.0
q10_kv4 = 3.0
q10_kv1_factor = q10_kv1 ** ((celsius - 22) / 10)
q10_kv4_factor = q10_kv4 ** ((celsius - 22) / 10)

# ==========================================
# Kv1 (LT_dth.mod)
# ==========================================

# Activation (o)
cao, kao = 0.6947, 0.03512
cbo, kbo = 0.02248, -0.0319
ao = cao * np.exp(kao * v)
bo = cbo * np.exp(kbo * v)
oinf = ao / (ao + bo)
otau = 1 / (ao + bo) / q10_kv1_factor

# Inactivation (p)
cap, kap = 0.00713, -0.1942
cbp, kbp = 0.0935, 0.0058
ap = cap * np.exp(kap * v)
bp = cbp * np.exp(kbp * v)
pinf = ap / (ap + bp)
ptau = 1 / (ap + bp) / q10_kv1_factor

# ==========================================
# Kv4 (klt.mod)
# ==========================================

# Activation (a)
ainf = (1 / (1 + np.exp(-(v + 40) / 6))) ** 0.25
atau = (100 / (7 * np.exp((v + 60) / 14) + 29 * np.exp(-(v + 60) / 24))) + 0.1
atau /= q10_kv4_factor

# Inactivation (b and c) — same expressions
binf = 1 / (1 + np.exp((v + 66) / 7)) ** 0.5
cinf = binf.copy()  # c has same expression as b
btau = (1000 / (14 * np.exp((v + 60) / 27) + 29 * np.exp(-(v + 60) / 24))) + 1
ctau = (90 / (1 + np.exp((-66 - v) / 17))) + 10
btau /= q10_kv4_factor
ctau /= q10_kv4_factor

# ==========================================
# === PLOTS
# ==========================================

# --- Activation (steady-state)
plt.figure(figsize=(8, 5))
plt.plot(v, oinf, label='Kv1 (oinf)', linewidth=2)
plt.plot(v, ainf, label='Kv4 (ainf)', linewidth=2, linestyle='--')
plt.xlabel('Membrane Voltage (mV)')
plt.ylabel('Steady-State Activation (∞)')
plt.title('Steady-State Activation at 35°C')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# --- Activation (time constant)
plt.figure(figsize=(8, 5))
plt.plot(v, otau, label='Kv1 (otau)', linewidth=2)
plt.plot(v, atau, label='Kv4 (atau)', linewidth=2, linestyle='--')
plt.xlabel('Membrane Voltage (mV)')
plt.ylabel('Activation Time Constant (ms)')
plt.title('Activation Time Constant at 35°C')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# --- Inactivation (steady-state)
plt.figure(figsize=(8, 5))
plt.plot(v, pinf, label='Kv1 (pinf)', linewidth=2)
plt.plot(v, binf, label='Kv4 (binf/cinf)', linewidth=2, linestyle='--')
plt.xlabel('Membrane Voltage (mV)')
plt.ylabel('Steady-State Inactivation (∞)')
plt.title('Steady-State Inactivation at 35°C')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# --- Inactivation (time constant)
plt.figure(figsize=(8, 5))
plt.plot(v, ptau, label='Kv1 (ptau)', linewidth=2)
plt.plot(v, btau, label='Kv4 (btau)', linewidth=2, linestyle='--')
plt.plot(v, ctau, label='Kv4 (ctau)', linewidth=2, linestyle=':')
plt.xlabel('Membrane Voltage (mV)')
plt.ylabel('Inactivation Time Constant (ms)')
plt.title('Inactivation Time Constants at 35°C')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
