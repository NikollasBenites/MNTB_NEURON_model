NEURON {
    SUFFIX NaCh_nmb
    USEION na READ ena WRITE ina
    RANGE gnabar, ina
    RANGE vhalf_m, k_m, vhalf_h, k_h
}

UNITS {
    (S) = (siemens)
    (mV) = (millivolt)
    (mA) = (milliamp)
}

PARAMETER {
    gnabar = 0.12 (S/cm2)
    ena = 50 (mV)

    vhalf_m = -35 (mV)
    k_m = -7.2 (mV)
    vhalf_h = -55 (mV)
    k_h = 5.0 (mV)

    taum = 0.1 (ms) : fixed time constants for simplicity
    tauh = 1.0 (ms)
}

STATE {
    m h
}

ASSIGNED {
    v (mV)
    ina (mA/cm2)
    minf hinf
}

BREAKPOINT {
    SOLVE states METHOD cnexp
    ina = gnabar * m * m * m * h * (v - ena)
}

INITIAL {
    rates(v)
    m = minf
    h = hinf
}

DERIVATIVE states {
    rates(v)
    m' = (minf - m)/taum
    h' = (hinf - h)/tauh
}

PROCEDURE rates(v(mV)) {
    minf = 1 / (1 + exp((v - vhalf_m)/k_m))
    hinf = 1 / (1 + exp((v - vhalf_h)/k_h))
}
