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
 
#define nrn_init _nrn_init__LT
#define _nrn_initial _nrn_initial__LT
#define nrn_cur _nrn_cur__LT
#define _nrn_current _nrn_current__LT
#define nrn_jacob _nrn_jacob__LT
#define nrn_state _nrn_state__LT
#define _net_receive _net_receive__LT 
#define rates rates__LT 
#define state state__LT 
 
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
#define gbar _p[0]
#define gbar_columnindex 0
#define ik _p[1]
#define ik_columnindex 1
#define l _p[2]
#define l_columnindex 2
#define r _p[3]
#define r_columnindex 3
#define ek _p[4]
#define ek_columnindex 4
#define Dl _p[5]
#define Dl_columnindex 5
#define Dr _p[6]
#define Dr_columnindex 6
#define _g _p[7]
#define _g_columnindex 7
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
 extern double celsius;
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
 "setdata_LT", _hoc_setdata,
 "rates_LT", _hoc_rates,
 0, 0
};
 /* declare global and static user variables */
#define ar ar_LT
 double ar = 0;
#define al al_LT
 double al = 0;
#define br br_LT
 double br = 0;
#define bl bl_LT
 double bl = 0;
#define ebr ebr_LT
 double ebr = -0.0047;
#define ear ear_LT
 double ear = -0.0053;
#define ebl ebl_LT
 double ebl = -0.0319;
#define eal eal_LT
 double eal = 0.03512;
#define gamma gamma_LT
 double gamma = 0.1;
#define kbr kbr_LT
 double kbr = 0.0562;
#define kar kar_LT
 double kar = 0.0438;
#define kbl kbl_LT
 double kbl = 0.2248;
#define kal kal_LT
 double kal = 1.2;
#define ltau ltau_LT
 double ltau = 0;
#define linf linf_LT
 double linf = 0;
#define rtau rtau_LT
 double rtau = 0;
#define rinf rinf_LT
 double rinf = 0;
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 0,0,0
};
 static HocParmUnits _hoc_parm_units[] = {
 "kal_LT", "/ms",
 "eal_LT", "/mV",
 "kbl_LT", "/ms",
 "ebl_LT", "/mV",
 "kar_LT", "/ms",
 "ear_LT", "/mV",
 "kbr_LT", "/ms",
 "ebr_LT", "/mV",
 "ltau_LT", "ms",
 "rtau_LT", "ms",
 "al_LT", "/ms",
 "bl_LT", "/ms",
 "ar_LT", "/ms",
 "br_LT", "/ms",
 "gbar_LT", "S/cm2",
 "ik_LT", "mA/cm2",
 0,0
};
 static double delta_t = 0.01;
 static double l0 = 0;
 static double r0 = 0;
 static double v = 0;
 /* connect global user variables to hoc */
 static DoubScal hoc_scdoub[] = {
 "gamma_LT", &gamma_LT,
 "kal_LT", &kal_LT,
 "eal_LT", &eal_LT,
 "kbl_LT", &kbl_LT,
 "ebl_LT", &ebl_LT,
 "kar_LT", &kar_LT,
 "ear_LT", &ear_LT,
 "kbr_LT", &kbr_LT,
 "ebr_LT", &ebr_LT,
 "linf_LT", &linf_LT,
 "ltau_LT", &ltau_LT,
 "rinf_LT", &rinf_LT,
 "rtau_LT", &rtau_LT,
 "al_LT", &al_LT,
 "bl_LT", &bl_LT,
 "ar_LT", &ar_LT,
 "br_LT", &br_LT,
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
"LT",
 "gbar_LT",
 0,
 "ik_LT",
 0,
 "l_LT",
 "r_LT",
 0,
 0};
 static Symbol* _k_sym;
 
extern Prop* need_memb(Symbol*);

static void nrn_alloc(Prop* _prop) {
	Prop *prop_ion;
	double *_p; Datum *_ppvar;
 	_p = nrn_prop_data_alloc(_mechtype, 8, _prop);
 	/*initialize range parameters*/
 	gbar = 0.002;
 	_prop->param = _p;
 	_prop->param_size = 8;
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

 void _lt_reg() {
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
  hoc_register_prop_size(_mechtype, 8, 4);
  hoc_register_dparam_semantics(_mechtype, 0, "k_ion");
  hoc_register_dparam_semantics(_mechtype, 1, "k_ion");
  hoc_register_dparam_semantics(_mechtype, 2, "k_ion");
  hoc_register_dparam_semantics(_mechtype, 3, "cvodeieq");
 	hoc_register_cvode(_mechtype, _ode_count, _ode_map, _ode_spec, _ode_matsol);
 	hoc_register_tolerance(_mechtype, _hoc_state_tol, &_atollist);
 	hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 LT lt.mod\n");
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
   Dl = ( linf - l ) / ltau ;
   Dr = ( rinf - r ) / rtau ;
   }
 return _reset;
}
 static int _ode_matsol1 () {
 rates ( _threadargscomma_ v ) ;
 Dl = Dl  / (1. - dt*( ( ( ( - 1.0 ) ) ) / ltau )) ;
 Dr = Dr  / (1. - dt*( ( ( ( - 1.0 ) ) ) / rtau )) ;
  return 0;
}
 /*END CVODE*/
 static int state () {_reset=0;
 {
   rates ( _threadargscomma_ v ) ;
    l = l + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / ltau)))*(- ( ( ( linf ) ) / ltau ) / ( ( ( ( - 1.0 ) ) ) / ltau ) - l) ;
    r = r + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / rtau)))*(- ( ( ( rinf ) ) / rtau ) / ( ( ( ( - 1.0 ) ) ) / rtau ) - r) ;
   }
  return 0;
}
 
static int  rates (  double _lv ) {
   al = kal * exp ( eal * _lv ) ;
   bl = kbl * exp ( ebl * _lv ) ;
   ar = kar * exp ( ear * _lv ) ;
   br = kbr * exp ( ebr * _lv ) ;
   linf = al / ( al + bl ) ;
   ltau = 1.0 / ( al + bl ) ;
   rinf = ar / ( ar + br ) ;
   rtau = 1.0 / ( ar + br ) ;
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
  l = l0;
  r = r0;
 {
   rates ( _threadargscomma_ v ) ;
   l = linf ;
   r = rinf ;
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
   ik = gbar * l * r * ( v - ek ) ;
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
 if(error){fprintf(stderr,"at line 69 in file lt.mod:\n	SOLVE state METHOD cnexp\n"); nrn_complain(_p); abort_run(error);}
 } }}

}

static void terminal(){}

static void _initlists() {
 int _i; static int _first = 1;
  if (!_first) return;
 _slist1[0] = l_columnindex;  _dlist1[0] = Dl_columnindex;
 _slist1[1] = r_columnindex;  _dlist1[1] = Dr_columnindex;
_first = 0;
}

#if NMODL_TEXT
static const char* nmodl_filename = "lt.mod";
static const char* nmodl_file_text = 
  "\n"
  ": 	Low threshold potassium chanel from\n"
  ":	Contribution of the Kv3.1 potassium channel to high-frequency firing in mouse auditory neurones\n"
  ":	Lu-Yang Wang, Li Gan, Ian D. Forsythe and Leonard K. Kaczmarek\n"
  ":	J. Physiol (1998), 501.9, pp. 183-194\n"
  "\n"
  "\n"
  "\n"
  "UNITS {\n"
  "	(mV) = (millivolt)\n"
  "	(S) = (mho)\n"
  "	(nA) = (nanoamp)\n"
  "}\n"
  "\n"
  "NEURON {\n"
  "	SUFFIX LT\n"
  "	USEION k READ ek WRITE ik\n"
  "	RANGE gbar, g, ik\n"
  "	GLOBAL linf, ltau, rinf, rtau, al, bl, ar, br\n"
  "}\n"
  "\n"
  ": area in paper is 1000 (um2) so divide our density parameters by 10\n"
  "\n"
  "PARAMETER {\n"
  "\n"
  "	celsius = 35 (degC)\n"
  "\n"
  "	gbar = .002 (S/cm2) : .02 (uS)\n"
  "	gamma = .1\n"
  "\n"
  "	kal = 1.2 (/ms)\n"
  "	eal = .03512 (/mV)\n"
  "	kbl = .2248 (/ms)\n"
  "	ebl = -.0319 (/mV)\n"
  "\n"
  "	kar = .0438 (/ms)\n"
  "	ear = -.0053 (/mV)\n"
  "	kbr = .0562 (/ms)\n"
  "	ebr = -.0047 (/mV)\n"
  "}\n"
  "\n"
  "ASSIGNED {\n"
  "	v (mV)\n"
  "	ek (mV)\n"
  "	ik (mA/cm2)\n"
  "\n"
  "	linf\n"
  "	ltau (ms)\n"
  "	rinf\n"
  "	rtau (ms)\n"
  "\n"
  "	al (/ms)\n"
  "	bl (/ms)\n"
  "	ar (/ms)\n"
  "	br (/ms)\n"
  "}\n"
  "\n"
  "STATE {\n"
  "	l r\n"
  "}\n"
  "\n"
  "INITIAL {\n"
  "	rates(v)\n"
  "	l = linf\n"
  "	r = rinf\n"
  "}\n"
  "\n"
  "BREAKPOINT {\n"
  "	SOLVE state METHOD cnexp\n"
  "	ik = gbar*l*r*(v - ek) : pemdas may be a problem\n"
  "}\n"
  "\n"
  "DERIVATIVE state {\n"
  "	rates(v)\n"
  "	l' = (linf - l)/ltau\n"
  "	r' = (rinf - r)/rtau\n"
  "}\n"
  "\n"
  "PROCEDURE rates(v(mV)) {\n"
  "	al = kal*exp(eal*v)\n"
  "	bl = kbl*exp(ebl*v)\n"
  "\n"
  "	ar = kar*exp(ear*v)\n"
  "	br = kbr*exp(ebr*v)\n"
  "\n"
  "	linf = al/(al + bl)\n"
  "	ltau = 1/(al + bl)\n"
  "	rinf = ar/(ar + br)\n"
  "	rtau = 1/(ar + br)\n"
  "}\n"
  "\n"
  ;
#endif
