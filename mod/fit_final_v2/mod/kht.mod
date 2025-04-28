: 	High threshold potassium channel from Sierksma et al., (2017)
:	and Wang et al., (1998)


NEURON {
	SUFFIX HT
	USEION k READ ek WRITE ik
	RANGE gkhtbar, gk, ik
	GLOBAL ninf, ntau, pinf, ptau, an, bn, ap, bp
}


UNITS {
	(mV) = (millivolt)
	(S) = (mho)
	(mA) = (milliamp)
}

PARAMETER {
	v (mV)
	ek (mV)
	gkhtbar = .015 (S/cm2)

	can = .2719 (/ms)
	kan = .04 (/mV)
	cbn = .1974 (/ms)
	kbn = 0 (/mV)

	cap = .00713 (/ms)
	kap = -.1942 (/mV)
	cbp = .0935 (/ms)
	kbp = .0058 (/mV)
}

ASSIGNED {
	ik (mA/cm2)
	gk (S/cm2)
	ninf
	ntau (ms)
	pinf
	ptau (ms)

	an (/ms)
	bn (/ms)
	ap (/ms)
	bp (/ms)
}

STATE {
	n p
}

INITIAL {
	rates(v)
	n = ninf
	p = pinf
}

BREAKPOINT {
	SOLVE state METHOD cnexp
	gk = gkhtbar*(n^3)*p
    ik = gk*(v - ek)
}

DERIVATIVE state {
	rates(v)
	n' = (ninf - n)/ntau
	p' = (pinf - p)/ptau
}

PROCEDURE rates(v(mV)) {
	an = can*exp(kan*v)
	bn = cbn*exp(kbn*v)

	ap = cap*exp(kap*v)
	bp = cbp*exp(kbp*v)

	ninf = an/(an + bn)
	ntau = 1/(an + bn)
	pinf = ap/(ap + bp)
	ptau = 1/(ap + bp)
}

