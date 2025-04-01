/* Created by Language version: 7.7.0 */
/* NOT VECTORIZED */
#define NRN_VECTORIZED 0
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mech_api.h"
#undef PI
#define nil 0
#include "md1redef.h"
#include "section.h"
#include "nrniv_mf.h"
#include "md2redef.h"
 
#if METHOD3
extern int _method3;
#endif

#if !NRNGPU
#undef exp
#define exp hoc_Exp
extern double hoc_Exp(double);
#endif
 
#define nrn_init _nrn_init__HT
#define _nrn_initial _nrn_initial__HT
#define nrn_cur _nrn_cur__HT
#define _nrn_current _nrn_current__HT
#define nrn_jacob _nrn_jacob__HT
#define nrn_state _nrn_state__HT
#define _net_receive _net_receive__HT 
#define rates rates__HT 
#define state state__HT 
 
#define _threadargscomma_ /**/
#define _threadargsprotocomma_ /**/
#define _threadargs_ /**/
#define _threadargsproto_ /**/
 	/*SUPPRESS 761*/
	/*SUPPRESS 762*/
	/*SUPPRESS 763*/
	/*SUPPRESS 765*/
	 extern double *getarg();
 static double *_p; static Datum *_ppvar;
 
#define t nrn_threads->_t
#define dt nrn_threads->_dt
#define gkhtbar _p[0]
#define gkhtbar_columnindex 0
#define ik _p[1]
#define ik_columnindex 1
#define gk _p[2]
#define gk_columnindex 2
#define n _p[3]
#define n_columnindex 3
#define p _p[4]
#define p_columnindex 4
#define ek _p[5]
#define ek_columnindex 5
#define Dn _p[6]
#define Dn_columnindex 6
#define Dp _p[7]
#define Dp_columnindex 7
#define _g _p[8]
#define _g_columnindex 8
#define _ion_ek	*_ppvar[0]._pval
#define _ion_ik	*_ppvar[1]._pval
#define _ion_dikdv	*_ppvar[2]._pval
 
#if MAC
#if !defined(v)
#define v _mlhv
#endif
#if !defined(h)
#define h _mlhh
#endif
#endif
 
#if defined(__cplusplus)
extern "C" {
#endif
 static int hoc_nrnpointerindex =  -1;
 /* external NEURON variables */
 /* declaration of user functions */
 static void _hoc_rates(void);
 static int _mechtype;
extern void _nrn_cacheloop_reg(int, int);
extern void hoc_register_prop_size(int, int, int);
extern void hoc_register_limits(int, HocParmLimits*);
extern void hoc_register_units(int, HocParmUnits*);
extern void nrn_promote(Prop*, int, int);
extern Memb_func* memb_func;
 
#define NMODL_TEXT 1
#if NMODL_TEXT
static const char* nmodl_file_text;
static const char* nmodl_filename;
extern void hoc_reg_nmodl_text(int, const char*);
extern void hoc_reg_nmodl_filename(int, const char*);
#endif

 extern void _nrn_setdata_reg(int, void(*)(Prop*));
 static void _setdata(Prop* _prop) {
 _p = _prop->param; _ppvar = _prop->dparam;
 }
 static void _hoc_setdata() {
 Prop *_prop, *hoc_getdata_range(int);
 _prop = hoc_getdata_range(_mechtype);
   _setdata(_prop);
 hoc_retpushx(1.);
}
 /* connect user functions to hoc names */
 static VoidFunc hoc_intfunc[] = {
 "setdata_HT", _hoc_setdata,
 "rates_HT", _hoc_rates,
 0, 0
};
 /* declare global and static user variables */
#define ap ap_HT
 double ap = 0;
#define an an_HT
 double an = 0;
#define bp bp_HT
 double bp = 0;
#define bn bn_HT
 double bn = 0;
#define cbp cbp_HT
 double cbp = 0.0935;
#define cap cap_HT
 double cap = 0.00713;
#define cbn cbn_HT
 double cbn = 0.1974;
#define can can_HT
 double can = 0.2719;
#define kbp kbp_HT
 double kbp = 0.0058;
#define kap kap_HT
 double kap = -0.1942;
#define kbn kbn_HT
 double kbn = 0;
#define kan kan_HT
 double kan = 0.04;
#define ntau ntau_HT
 double ntau = 0;
#define ninf ninf_HT
 double ninf = 0;
#define ptau ptau_HT
 double ptau = 0;
#define pinf pinf_HT
 double pinf = 0;
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 0,0,0
};
 static HocParmUnits _hoc_parm_units[] = {
 "can_HT", "/ms",
 "kan_HT", "/mV",
 "cbn_HT", "/ms",
 "kbn_HT", "/mV",
 "cap_HT", "/ms",
 "kap_HT", "/mV",
 "cbp_HT", "/ms",
 "kbp_HT", "/mV",
 "ntau_HT", "ms",
 "ptau_HT", "ms",
 "an_HT", "/ms",
 "bn_HT", "/ms",
 "ap_HT", "/ms",
 "bp_HT", "/ms",
 "gkhtbar_HT", "S/cm2",
 "ik_HT", "mA/cm2",
 "gk_HT", "S/cm2",
 0,0
};
 static double delta_t = 0.01;
 static double n0 = 0;
 static double p0 = 0;
 static double v = 0;
 /* connect global user variables to hoc */
 static DoubScal hoc_scdoub[] = {
 "can_HT", &can_HT,
 "kan_HT", &kan_HT,
 "cbn_HT", &cbn_HT,
 "kbn_HT", &kbn_HT,
 "cap_HT", &cap_HT,
 "kap_HT", &kap_HT,
 "cbp_HT", &cbp_HT,
 "kbp_HT", &kbp_HT,
 "ninf_HT", &ninf_HT,
 "ntau_HT", &ntau_HT,
 "pinf_HT", &pinf_HT,
 "ptau_HT", &ptau_HT,
 "an_HT", &an_HT,
 "bn_HT", &bn_HT,
 "ap_HT", &ap_HT,
 "bp_HT", &bp_HT,
 0,0
};
 static DoubVec hoc_vdoub[] = {
 0,0,0
};
 static double _sav_indep;
 static void nrn_alloc(Prop*);
static void  nrn_init(NrnThread*, _Memb_list*, int);
static void nrn_state(NrnThread*, _Memb_list*, int);
 static void nrn_cur(NrnThread*, _Memb_list*, int);
static void  nrn_jacob(NrnThread*, _Memb_list*, int);
 
static int _ode_count(int);
static void _ode_map(int, double**, double**, double*, Datum*, double*, int);
static void _ode_spec(NrnThread*, _Memb_list*, int);
static void _ode_matsol(NrnThread*, _Memb_list*, int);
 
#define _cvode_ieq _ppvar[3]._i
 static void _ode_matsol_instance1(_threadargsproto_);
 /* connect range variables in _p that hoc is supposed to know about */
 static const char *_mechanism[] = {
 "7.7.0",
"HT",
 "gkhtbar_HT",
 0,
 "ik_HT",
 "gk_HT",
 0,
 "n_HT",
 "p_HT",
 0,
 0};
 static Symbol* _k_sym;
 
extern Prop* need_memb(Symbol*);

static void nrn_alloc(Prop* _prop) {
	Prop *prop_ion;
	double *_p; Datum *_ppvar;
 	_p = nrn_prop_data_alloc(_mechtype, 9, _prop);
 	/*initialize range parameters*/
 	gkhtbar = 0.015;
 	_prop->param = _p;
 	_prop->param_size = 9;
 	_ppvar = nrn_prop_datum_alloc(_mechtype, 4, _prop);
 	_prop->dparam = _ppvar;
 	/*connect ionic variables to this model*/
 prop_ion = need_memb(_k_sym);
 nrn_promote(prop_ion, 0, 1);
 	_ppvar[0]._pval = &prop_ion->param[0]; /* ek */
 	_ppvar[1]._pval = &prop_ion->param[3]; /* ik */
 	_ppvar[2]._pval = &prop_ion->param[4]; /* _ion_dikdv */
 
}
 static void _initlists();
  /* some states have an absolute tolerance */
 static Symbol** _atollist;
 static HocStateTolerance _hoc_state_tol[] = {
 0,0
};
 static void _update_ion_pointer(Datum*);
 extern Symbol* hoc_lookup(const char*);
extern void _nrn_thread_reg(int, int, void(*)(Datum*));
extern void _nrn_thread_table_reg(int, void(*)(double*, Datum*, Datum*, NrnThread*, int));
extern void hoc_register_tolerance(int, HocStateTolerance*, Symbol***);
extern void _cvode_abstol( Symbol**, double*, int);

 void _kht_reg() {
	int _vectorized = 0;
  _initlists();
 	ion_reg("k", -10000.);
 	_k_sym = hoc_lookup("k_ion");
 	register_mech(_mechanism, nrn_alloc,nrn_cur, nrn_jacob, nrn_state, nrn_init, hoc_nrnpointerindex, 0);
 _mechtype = nrn_get_mechtype(_mechanism[1]);
     _nrn_setdata_reg(_mechtype, _setdata);
     _nrn_thread_reg(_mechtype, 2, _update_ion_pointer);
 #if NMODL_TEXT
  hoc_reg_nmodl_text(_mechtype, nmodl_file_text);
  hoc_reg_nmodl_filename(_mechtype, nmodl_filename);
#endif
  hoc_register_prop_size(_mechtype, 9, 4);
  hoc_register_dparam_semantics(_mechtype, 0, "k_ion");
  hoc_register_dparam_semantics(_mechtype, 1, "k_ion");
  hoc_register_dparam_semantics(_mechtype, 2, "k_ion");
  hoc_register_dparam_semantics(_mechtype, 3, "cvodeieq");
 	hoc_register_cvode(_mechtype, _ode_count, _ode_map, _ode_spec, _ode_matsol);
 	hoc_register_tolerance(_mechtype, _hoc_state_tol, &_atollist);
 	hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 HT /Users/nikollas/Library/CloudStorage/OneDrive-UniversityofSouthFlorida/MNTB_neuron/mod/kht.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
static int _reset;
static char *modelname = "";

static int error;
static int _ninits = 0;
static int _match_recurse=1;
static void _modl_cleanup(){ _match_recurse=1;}
static int rates(double);
 
static int _ode_spec1(_threadargsproto_);
/*static int _ode_matsol1(_threadargsproto_);*/
 static int _slist1[2], _dlist1[2];
 static int state(_threadargsproto_);
 
/*CVODE*/
 static int _ode_spec1 () {_reset=0;
 {
   rates ( _threadargscomma_ v ) ;
   Dn = ( ninf - n ) / ntau ;
   Dp = ( pinf - p ) / ptau ;
   }
 return _reset;
}
 static int _ode_matsol1 () {
 rates ( _threadargscomma_ v ) ;
 Dn = Dn  / (1. - dt*( ( ( ( - 1.0 ) ) ) / ntau )) ;
 Dp = Dp  / (1. - dt*( ( ( ( - 1.0 ) ) ) / ptau )) ;
  return 0;
}
 /*END CVODE*/
 static int state () {_reset=0;
 {
   rates ( _threadargscomma_ v ) ;
    n = n + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / ntau)))*(- ( ( ( ninf ) ) / ntau ) / ( ( ( ( - 1.0 ) ) ) / ntau ) - n) ;
    p = p + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / ptau)))*(- ( ( ( pinf ) ) / ptau ) / ( ( ( ( - 1.0 ) ) ) / ptau ) - p) ;
   }
  return 0;
}
 
static int  rates (  double _lv ) {
   an = can * exp ( kan * _lv ) ;
   bn = cbn * exp ( kbn * _lv ) ;
   ap = cap * exp ( kap * _lv ) ;
   bp = cbp * exp ( kbp * _lv ) ;
   ninf = an / ( an + bn ) ;
   ntau = 1.0 / ( an + bn ) ;
   pinf = ap / ( ap + bp ) ;
   ptau = 1.0 / ( ap + bp ) ;
    return 0; }
 
static void _hoc_rates(void) {
  double _r;
   _r = 1.;
 rates (  *getarg(1) );
 hoc_retpushx(_r);
}
 
static int _ode_count(int _type){ return 2;}
 
static void _ode_spec(NrnThread* _nt, _Memb_list* _ml, int _type) {
   Datum* _thread;
   Node* _nd; double _v; int _iml, _cntml;
  _cntml = _ml->_nodecount;
  _thread = _ml->_thread;
  for (_iml = 0; _iml < _cntml; ++_iml) {
    _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
    _nd = _ml->_nodelist[_iml];
    v = NODEV(_nd);
  ek = _ion_ek;
     _ode_spec1 ();
  }}
 
static void _ode_map(int _ieq, double** _pv, double** _pvdot, double* _pp, Datum* _ppd, double* _atol, int _type) { 
 	int _i; _p = _pp; _ppvar = _ppd;
	_cvode_ieq = _ieq;
	for (_i=0; _i < 2; ++_i) {
		_pv[_i] = _pp + _slist1[_i];  _pvdot[_i] = _pp + _dlist1[_i];
		_cvode_abstol(_atollist, _atol, _i);
	}
 }
 
static void _ode_matsol_instance1(_threadargsproto_) {
 _ode_matsol1 ();
 }
 
static void _ode_matsol(NrnThread* _nt, _Memb_list* _ml, int _type) {
   Datum* _thread;
   Node* _nd; double _v; int _iml, _cntml;
  _cntml = _ml->_nodecount;
  _thread = _ml->_thread;
  for (_iml = 0; _iml < _cntml; ++_iml) {
    _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
    _nd = _ml->_nodelist[_iml];
    v = NODEV(_nd);
  ek = _ion_ek;
 _ode_matsol_instance1(_threadargs_);
 }}
 extern void nrn_update_ion_pointer(Symbol*, Datum*, int, int);
 static void _update_ion_pointer(Datum* _ppvar) {
   nrn_update_ion_pointer(_k_sym, _ppvar, 0, 0);
   nrn_update_ion_pointer(_k_sym, _ppvar, 1, 3);
   nrn_update_ion_pointer(_k_sym, _ppvar, 2, 4);
 }

static void initmodel() {
  int _i; double _save;_ninits++;
 _save = t;
 t = 0.0;
{
  n = n0;
  p = p0;
 {
   rates ( _threadargscomma_ v ) ;
   n = ninf ;
   p = pinf ;
   }
  _sav_indep = t; t = _save;

}
}

static void nrn_init(NrnThread* _nt, _Memb_list* _ml, int _type){
Node *_nd; double _v; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
 v = _v;
  ek = _ion_ek;
 initmodel();
 }}

static double _nrn_current(double _v){double _current=0.;v=_v;{ {
   gk = gkhtbar * ( pow( n , 3.0 ) ) * p ;
   ik = gk * ( v - ek ) ;
   }
 _current += ik;

} return _current;
}

static void nrn_cur(NrnThread* _nt, _Memb_list* _ml, int _type){
Node *_nd; int* _ni; double _rhs, _v; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
  ek = _ion_ek;
 _g = _nrn_current(_v + .001);
 	{ double _dik;
  _dik = ik;
 _rhs = _nrn_current(_v);
  _ion_dikdv += (_dik - ik)/.001 ;
 	}
 _g = (_g - _rhs)/.001;
  _ion_ik += ik ;
#if CACHEVEC
  if (use_cachevec) {
	VEC_RHS(_ni[_iml]) -= _rhs;
  }else
#endif
  {
	NODERHS(_nd) -= _rhs;
  }
 
}}

static void nrn_jacob(NrnThread* _nt, _Memb_list* _ml, int _type){
Node *_nd; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml];
#if CACHEVEC
  if (use_cachevec) {
	VEC_D(_ni[_iml]) += _g;
  }else
#endif
  {
     _nd = _ml->_nodelist[_iml];
	NODED(_nd) += _g;
  }
 
}}

static void nrn_state(NrnThread* _nt, _Memb_list* _ml, int _type){
Node *_nd; double _v = 0.0; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
 _nd = _ml->_nodelist[_iml];
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
 v=_v;
{
  ek = _ion_ek;
 { error =  state();
 if(error){fprintf(stderr,"at line 60 in file kht.mod:\n	SOLVE state METHOD cnexp\n"); nrn_complain(_p); abort_run(error);}
 } }}

}

static void terminal(){}

static void _initlists() {
 int _i; static int _first = 1;
  if (!_first) return;
 _slist1[0] = n_columnindex;  _dlist1[0] = Dn_columnindex;
 _slist1[1] = p_columnindex;  _dlist1[1] = Dp_columnindex;
_first = 0;
}

#if NMODL_TEXT
static const char* nmodl_filename = "/Users/nikollas/Library/CloudStorage/OneDrive-UniversityofSouthFlorida/MNTB_neuron/mod/kht.mod";
static const char* nmodl_file_text = 
  ": 	High threshold potassium channel from Sierksma et al., (2017)\n"
  ":	and Wang et al., (1998)\n"
  "\n"
  "\n"
  "NEURON {\n"
  "	SUFFIX HT\n"
  "	USEION k READ ek WRITE ik\n"
  "	RANGE gkhtbar, gk, ik\n"
  "	GLOBAL ninf, ntau, pinf, ptau, an, bn, ap, bp\n"
  "}\n"
  "\n"
  "\n"
  "UNITS {\n"
  "	(mV) = (millivolt)\n"
  "	(S) = (mho)\n"
  "	(mA) = (milliamp)\n"
  "}\n"
  "\n"
  "PARAMETER {\n"
  "	v (mV)\n"
  "	ek (mV)\n"
  "	gkhtbar = .015 (S/cm2)\n"
  "\n"
  "	can = .2719 (/ms)\n"
  "	kan = .04 (/mV)\n"
  "	cbn = .1974 (/ms)\n"
  "	kbn = 0 (/mV)\n"
  "\n"
  "	cap = .00713 (/ms)\n"
  "	kap = -.1942 (/mV)\n"
  "	cbp = .0935 (/ms)\n"
  "	kbp = .0058 (/mV)\n"
  "}\n"
  "\n"
  "ASSIGNED {\n"
  "	ik (mA/cm2)\n"
  "	gk (S/cm2)\n"
  "	ninf\n"
  "	ntau (ms)\n"
  "	pinf\n"
  "	ptau (ms)\n"
  "\n"
  "	an (/ms)\n"
  "	bn (/ms)\n"
  "	ap (/ms)\n"
  "	bp (/ms)\n"
  "}\n"
  "\n"
  "STATE {\n"
  "	n p\n"
  "}\n"
  "\n"
  "INITIAL {\n"
  "	rates(v)\n"
  "	n = ninf\n"
  "	p = pinf\n"
  "}\n"
  "\n"
  "BREAKPOINT {\n"
  "	SOLVE state METHOD cnexp\n"
  "	gk = gkhtbar*(n^3)*p\n"
  "    ik = gk*(v - ek)\n"
  "}\n"
  "\n"
  "DERIVATIVE state {\n"
  "	rates(v)\n"
  "	n' = (ninf - n)/ntau\n"
  "	p' = (pinf - p)/ptau\n"
  "}\n"
  "\n"
  "PROCEDURE rates(v(mV)) {\n"
  "	an = can*exp(kan*v)\n"
  "	bn = cbn*exp(kbn*v)\n"
  "\n"
  "	ap = cap*exp(kap*v)\n"
  "	bp = cbp*exp(kbp*v)\n"
  "\n"
  "	ninf = an/(an + bn)\n"
  "	ntau = 1/(an + bn)\n"
  "	pinf = ap/(ap + bp)\n"
  "	ptau = 1/(ap + bp)\n"
  "}\n"
  "\n"
  ;
#endif
