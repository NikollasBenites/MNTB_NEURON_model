: 	Low threshold potassium channel from Sierksma et al., (2017)
:	and Wang et al., (1998)

NEURON {
	SUFFIX LT_nmb
	USEION k READ ek WRITE ik
	RANGE gkltbar, gk, ik
	GLOBAL oinf, otau, ao, bo
}


UNITS {
	(mV) = (millivolt)
	(S) = (mho)
	(mA) = (milliamp)
}

PARAMETER {
	v (mV)
	ek (mV)
	gkltbar = .002 (S/cm2)

	cao = 6.947 (/ms)
	kao = .03512 (/mV)
	cbo = .2248 (/ms)
	kbo = -.0319 (/mV)

}

ASSIGNED {
	ik (mA/cm2)
	gk (S/cm2)

	oinf
	otau (ms)

	ao (/ms)
	bo (/ms)
}

STATE {
	o
}

INITIAL {
	rates(v)
	o = oinf
}

BREAKPOINT {
	SOLVE state METHOD cnexp
	gk = gkltbar*(o^3)
    ik = gk*(v - ek)
}

DERIVATIVE state {
	rates(v)
	o' = (oinf - o)/otau
}

PROCEDURE rates(v(mV)) {
	ao = cao*exp(kao*v)
	bo = cbo*exp(kbo*v)

	oinf = ao/(ao + bo)
	otau = 1/(ao + bo)
}

