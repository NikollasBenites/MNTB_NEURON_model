: 	Hyperpolarization activated current (Ih) Sierksma et al., (2017)
:	and Wang et al., (1998)


NEURON {
	SUFFIX IH
	NONSPECIFIC_CURRENT i
    RANGE ghbar, gh, ih, eh
    GLOBAL uinf, utau, au, bu
}


UNITS {
	(mV) = (millivolt)
	(S) = (mho)
	(mA) = (milliamp)
}

PARAMETER {
	v (mV)
	ghbar = .0037 (S/cm2)
	eh (mV)

	cau = 9.12e-8 (/ms)
	kau = -0.1 (/mV)
	cbu = .0021 (/ms)
	kbu = 0 (/mV)

}

ASSIGNED {
	gh (S/cm2)
	i (mA/cm2)

	uinf
	utau (ms)

	au (/ms)
	bu (/ms)
}

STATE {
	u
}

INITIAL {
	rates(v)
	u = uinf
}

BREAKPOINT {
	SOLVE state METHOD cnexp
	gh = ghbar*u
	i = gh*(v - eh)
}

DERIVATIVE state {
	rates(v)
	u' = (uinf - u)/utau
}

PROCEDURE rates(v(mV)) {
	au = cau*exp(kau*v)
	bu = cbu*exp(kbu*v)

	uinf = au/(au + bu)
	utau = 1/(au + bu)

}

