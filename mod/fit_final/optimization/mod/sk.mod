NEURON {
    SUFFIX sk
    USEION k READ ek WRITE ik
    USEION ca READ cai
    RANGE gbar, ik
    RANGE m_inf, tau_m
    GLOBAL ca_half, k_ca
}
UNITS {
    (molar) = (1/liter)
    (mM) = (millimolar)
}

PARAMETER {
    gbar = 0.001 (S/cm2)
    ek (mV)
    cai (mM)

    ca_half = 0.0005 (mM)    : half activation at 0.5 uM
    k_ca = 0.00025 (mM)      : Ca sensitivity (Hill-type)

    tau_m = 30 (ms)          : slower activation
}


STATE {
    m
}

ASSIGNED {
    v (mV)
    ik (mA/cm2)
    m_inf
}

INITIAL {
    rates(cai)
    m = m_inf
}

BREAKPOINT {
    SOLVE states METHOD cnexp
    ik = gbar * m * (v - ek)
}

DERIVATIVE states {
    rates(cai)
    m' = (m_inf - m) / tau_m
}

PROCEDURE rates(cai (mM)) {
    : Hill equation
    m_inf = cai^4 / (cai^4 + ca_half^4)
}
