NEURON {
    SUFFIX NaCh_nmb
    USEION na READ ena WRITE ina
    RANGE gnabar, ina
    RANGE cam, kam, cbm, kbm
    RANGE cah, kah, cbh, kbh
}

UNITS {
    (S) = (siemens)
    (mV) = (millivolt)
    (mA) = (milliamp)
}

PARAMETER {
    v (mV)
    gnabar = 0.05 (S/cm2)
    ena (mV)

    cam = 76.4 (/ms)
    kam = 0.037 (/mV)
    cbm = 6.930852 (/ms)
    kbm = -0.043 (/mV)

    cah = 0.000533 (/ms)
    kah = -0.0909 (/mV)
    cbh = 0.787 (/ms)
    kbh = 0.0691 (/mV)
}

ASSIGNED {
	ina (mA/cm2)
	gna (S/cm2)
	minf
	mtau (ms)
	hinf
	htau (ms)

	am (/ms)
	bm (/ms)
	ah (/ms)
	bh (/ms)
}

STATE {
    m h
}

INITIAL {
    rates(v)
    m = minf
    h = hinf
}

BREAKPOINT {
    SOLVE state METHOD cnexp
    gna = gnabar*(m^3)*h
    ina = gna*(v - ena)
}

DERIVATIVE state {
    rates(v)
    m' = (minf - m)/mtau
    h' = (hinf - h)/htau
}

PROCEDURE rates(v (mV)) {
    LOCAL am, bm, ah, bh

    am = cam * exp(kam * v)
    bm = cbm * exp(kbm * v)
    ah = cah * exp(kah * v)
    bh = cbh * exp(kbh * v)

    mtau = 1 / (am + bm)
    minf = am / (am + bm)

    htau = 1 / (ah + bh)
    hinf = ah / (ah + bh)
}
