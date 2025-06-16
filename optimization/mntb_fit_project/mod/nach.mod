: 	Sodium channel modeled from Sierksma et al., (2017)
:	and Wang et al., (1998)

NEURON {
	SUFFIX NaCh
	USEION na READ ena WRITE ina
	RANGE gnabar, gna, ina
	GLOBAL minf, mtau, hinf, htau, am, bm, ah, bh
}


UNITS {
	(mV) = (millivolt)
	(S) = (mho)
	(mA) = (milliamp)
}

PARAMETER {
	v (mV)
	ena (mV)
	gnabar = .05 (S/cm2)

	cam = 76.4 (/ms)
	kam = .037 (/mV)
	cbm = 6.930852 (/ms)
	kbm = -.043 (/mV)

	cah = 0.000533 (/ms)
	kah = -.0909 (/mV)
	cbh = .787 (/ms)
	kbh = .0691 (/mV)
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

PROCEDURE rates(v(mV)) {
	am = cam*exp(kam*v)
	bm = cbm*exp(kbm*v)

	ah = cah*exp(kah*v)
	bh = cbh*exp(kbh*v)

	minf = am/(am + bm)
	mtau = 1/(am + bm)
	hinf = ah/(ah + bh)
	htau = 1/(ah + bh)
}

