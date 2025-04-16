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
 
#define nrn_init _nrn_init__NaCh_dth
#define _nrn_initial _nrn_initial__NaCh_dth
#define nrn_cur _nrn_cur__NaCh_dth
#define _nrn_current _nrn_current__NaCh_dth
#define nrn_jacob _nrn_jacob__NaCh_dth
#define nrn_state _nrn_state__NaCh_dth
#define _net_receive _net_receive__NaCh_dth 
#define rates rates__NaCh_dth 
#define state state__NaCh_dth 
 
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
#define gnabar _p[0]
#define gnabar_columnindex 0
#define ina _p[1]
#define ina_columnindex 1
#define gna _p[2]
#define gna_columnindex 2
#define m _p[3]
#define m_columnindex 3
#define h _p[4]
#define h_columnindex 4
#define ena _p[5]
#define ena_columnindex 5
#define qg _p[6]
#define qg_columnindex 6
#define q10 _p[7]
#define q10_columnindex 7
#define Dm _p[8]
#define Dm_columnindex 8
#define Dh _p[9]
#define Dh_columnindex 9
#define _g _p[10]
#define _g_columnindex 10
#define _ion_ena	*_ppvar[0]._pval
#define _ion_ina	*_ppvar[1]._pval
#define _ion_dinadv	*_ppvar[2]._pval
 
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
 "setdata_NaCh_dth", _hoc_setdata,
 "rates_NaCh_dth", _hoc_rates,
 0, 0
};
 /* declare global and static user variables */
#define ah ah_NaCh_dth
 double ah = 0;
#define am am_NaCh_dth
 double am = 0;
#define bh bh_NaCh_dth
 double bh = 0;
#define bm bm_NaCh_dth
 double bm = 0;
#define cbh cbh_NaCh_dth
 double cbh = 0.787;
#define cah cah_NaCh_dth
 double cah = 0.000533;
#define cbm cbm_NaCh_dth
 double cbm = 6.93085;
#define cam cam_NaCh_dth
 double cam = 76.4;
#define htau htau_NaCh_dth
 double htau = 0;
#define hinf hinf_NaCh_dth
 double hinf = 0;
#define kbh kbh_NaCh_dth
 double kbh = 0.0691;
#define kah kah_NaCh_dth
 double kah = -0.0909;
#define kbm kbm_NaCh_dth
 double kbm = -0.043;
#define kam kam_NaCh_dth
 double kam = 0.037;
#define mtau mtau_NaCh_dth
 double mtau = 0;
#define minf minf_NaCh_dth
 double minf = 0;
#define q10g q10g_NaCh_dth
 double q10g = 2;
#define q10tau q10tau_NaCh_dth
 double q10tau = 3;
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 0,0,0
};
 static HocParmUnits _hoc_parm_units[] = {
 "cam_NaCh_dth", "/ms",
 "kam_NaCh_dth", "/mV",
 "cbm_NaCh_dth", "/ms",
 "kbm_NaCh_dth", "/mV",
 "cah_NaCh_dth", "/ms",
 "kah_NaCh_dth", "/mV",
 "cbh_NaCh_dth", "/ms",
 "kbh_NaCh_dth", "/mV",
 "mtau_NaCh_dth", "ms",
 "htau_NaCh_dth", "ms",
 "am_NaCh_dth", "/ms",
 "bm_NaCh_dth", "/ms",
 "ah_NaCh_dth", "/ms",
 "bh_NaCh_dth", "/ms",
 "gnabar_NaCh_dth", "S/cm2",
 "ina_NaCh_dth", "mA/cm2",
 "gna_NaCh_dth", "S/cm2",
 0,0
};
 static double delta_t = 0.01;
 static double h0 = 0;
 static double m0 = 0;
 static double v = 0;
 /* connect global user variables to hoc */
 static DoubScal hoc_scdoub[] = {
 "q10tau_NaCh_dth", &q10tau_NaCh_dth,
 "q10g_NaCh_dth", &q10g_NaCh_dth,
 "cam_NaCh_dth", &cam_NaCh_dth,
 "kam_NaCh_dth", &kam_NaCh_dth,
 "cbm_NaCh_dth", &cbm_NaCh_dth,
 "kbm_NaCh_dth", &kbm_NaCh_dth,
 "cah_NaCh_dth", &cah_NaCh_dth,
 "kah_NaCh_dth", &kah_NaCh_dth,
 "cbh_NaCh_dth", &cbh_NaCh_dth,
 "kbh_NaCh_dth", &kbh_NaCh_dth,
 "minf_NaCh_dth", &minf_NaCh_dth,
 "mtau_NaCh_dth", &mtau_NaCh_dth,
 "hinf_NaCh_dth", &hinf_NaCh_dth,
 "htau_NaCh_dth", &htau_NaCh_dth,
 "am_NaCh_dth", &am_NaCh_dth,
 "bm_NaCh_dth", &bm_NaCh_dth,
 "ah_NaCh_dth", &ah_NaCh_dth,
 "bh_NaCh_dth", &bh_NaCh_dth,
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
"NaCh_dth",
 "gnabar_NaCh_dth",
 0,
 "ina_NaCh_dth",
 "gna_NaCh_dth",
 0,
 "m_NaCh_dth",
 "h_NaCh_dth",
 0,
 0};
 static Symbol* _na_sym;
 
extern Prop* need_memb(Symbol*);

static void nrn_alloc(Prop* _prop) {
	Prop *prop_ion;
	double *_p; Datum *_ppvar;
 	_p = nrn_prop_data_alloc(_mechtype, 11, _prop);
 	/*initialize range parameters*/
 	gnabar = 0.05;
 	_prop->param = _p;
 	_prop->param_size = 11;
 	_ppvar = nrn_prop_datum_alloc(_mechtype, 4, _prop);
 	_prop->dparam = _ppvar;
 	/*connect ionic variables to this model*/
 prop_ion = need_memb(_na_sym);
 nrn_promote(prop_ion, 0, 1);
 	_ppvar[0]._pval = &prop_ion->param[0]; /* ena */
 	_ppvar[1]._pval = &prop_ion->param[3]; /* ina */
 	_ppvar[2]._pval = &prop_ion->param[4]; /* _ion_dinadv */
 
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

 void _nach_dth_reg() {
	int _vectorized = 0;
  _initlists();
 	ion_reg("na", -10000.);
 	_na_sym = hoc_lookup("na_ion");
 	register_mech(_mechanism, nrn_alloc,nrn_cur, nrn_jacob, nrn_state, nrn_init, hoc_nrnpointerindex, 0);
 _mechtype = nrn_get_mechtype(_mechanism[1]);
     _nrn_setdata_reg(_mechtype, _setdata);
     _nrn_thread_reg(_mechtype, 2, _update_ion_pointer);
 #if NMODL_TEXT
  hoc_reg_nmodl_text(_mechtype, nmodl_file_text);
  hoc_reg_nmodl_filename(_mechtype, nmodl_filename);
#endif
  hoc_register_prop_size(_mechtype, 11, 4);
  hoc_register_dparam_semantics(_mechtype, 0, "na_ion");
  hoc_register_dparam_semantics(_mechtype, 1, "na_ion");
  hoc_register_dparam_semantics(_mechtype, 2, "na_ion");
  hoc_register_dparam_semantics(_mechtype, 3, "cvodeieq");
 	hoc_register_cvode(_mechtype, _ode_count, _ode_map, _ode_spec, _ode_matsol);
 	hoc_register_tolerance(_mechtype, _hoc_state_tol, &_atollist);
 	hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 NaCh_dth /Users/nikollas/Library/CloudStorage/OneDrive-UniversityofSouthFlorida/MNTB_neuron/MNTB_Model_Dann/nach_dth.mod\n");
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
   Dm = ( minf - m ) / mtau ;
   Dh = ( hinf - h ) / htau ;
   }
 return _reset;
}
 static int _ode_matsol1 () {
 rates ( _threadargscomma_ v ) ;
 Dm = Dm  / (1. - dt*( ( ( ( - 1.0 ) ) ) / mtau )) ;
 Dh = Dh  / (1. - dt*( ( ( ( - 1.0 ) ) ) / htau )) ;
  return 0;
}
 /*END CVODE*/
 static int state () {_reset=0;
 {
   rates ( _threadargscomma_ v ) ;
    m = m + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / mtau)))*(- ( ( ( minf ) ) / mtau ) / ( ( ( ( - 1.0 ) ) ) / mtau ) - m) ;
    h = h + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / htau)))*(- ( ( ( hinf ) ) / htau ) / ( ( ( ( - 1.0 ) ) ) / htau ) - h) ;
   }
  return 0;
}
 
static int  rates (  double _lv ) {
   am = cam * exp ( kam * _lv ) ;
   bm = cbm * exp ( kbm * _lv ) ;
   ah = cah * exp ( kah * _lv ) ;
   bh = cbh * exp ( kbh * _lv ) ;
   minf = am / ( am + bm ) ;
   mtau = 1.0 / ( am + bm ) ;
   mtau = mtau / q10 ;
   hinf = ah / ( ah + bh ) ;
   htau = 1.0 / ( ah + bh ) ;
   htau = htau / q10 ;
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
  ena = _ion_ena;
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
  ena = _ion_ena;
 _ode_matsol_instance1(_threadargs_);
 }}
 extern void nrn_update_ion_pointer(Symbol*, Datum*, int, int);
 static void _update_ion_pointer(Datum* _ppvar) {
   nrn_update_ion_pointer(_na_sym, _ppvar, 0, 0);
   nrn_update_ion_pointer(_na_sym, _ppvar, 1, 3);
   nrn_update_ion_pointer(_na_sym, _ppvar, 2, 4);
 }

static void initmodel() {
  int _i; double _save;_ninits++;
 _save = t;
 t = 0.0;
{
  h = h0;
  m = m0;
 {
   qg = pow( q10g , ( ( celsius - 22.0 ) / 10.0 ) ) ;
   q10 = pow( q10tau , ( ( celsius - 22.0 ) / 10.0 ) ) ;
   rates ( _threadargscomma_ v ) ;
   m = minf ;
   h = hinf ;
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
  ena = _ion_ena;
 initmodel();
 }}

static double _nrn_current(double _v){double _current=0.;v=_v;{ {
   gna = qg * gnabar * ( pow( m , 3.0 ) ) * h ;
   ina = gna * ( v - ena ) ;
   }
 _current += ina;

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
  ena = _ion_ena;
 _g = _nrn_current(_v + .001);
 	{ double _dina;
  _dina = ina;
 _rhs = _nrn_current(_v);
  _ion_dinadv += (_dina - ina)/.001 ;
 	}
 _g = (_g - _rhs)/.001;
  _ion_ina += ina ;
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
  ena = _ion_ena;
 { error =  state();
 if(error){fprintf(stderr,"at line 66 in file nach_dth.mod:\n	SOLVE state METHOD cnexp\n"); nrn_complain(_p); abort_run(error);}
 } }}

}

static void terminal(){}

static void _initlists() {
 int _i; static int _first = 1;
  if (!_first) return;
 _slist1[0] = m_columnindex;  _dlist1[0] = Dm_columnindex;
 _slist1[1] = h_columnindex;  _dlist1[1] = Dh_columnindex;
_first = 0;
}

#if NMODL_TEXT
static const char* nmodl_filename = "/Users/nikollas/Library/CloudStorage/OneDrive-UniversityofSouthFlorida/MNTB_neuron/MNTB_Model_Dann/nach_dth.mod";
static const char* nmodl_file_text = 
  ": 	Sodium channel modeled from Sierksma et al., (2017)\n"
  ":	and Wang et al., (1998)\n"
  "\n"
  "NEURON {\n"
  "	SUFFIX NaCh_dth\n"
  "	USEION na READ ena WRITE ina\n"
  "	RANGE gnabar, gna, ina\n"
  "	GLOBAL minf, mtau, hinf, htau, am, bm, ah, bh\n"
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
  "	ena (mV)\n"
  "	gnabar = .05 (S/cm2)\n"
  "	q10tau = 3.0\n"
  "	q10g = 2.0\n"
  "\n"
  "	cam = 76.4 (/ms)\n"
  "	kam = .037 (/mV)\n"
  "	cbm = 6.930852 (/ms)\n"
  "	kbm = -.043 (/mV)\n"
  "\n"
  "	cah = 0.000533 (/ms)\n"
  "	kah = -.0909 (/mV)\n"
  "	cbh = .787 (/ms)\n"
  "	kbh = .0691 (/mV)\n"
  "}\n"
  "\n"
  "ASSIGNED {\n"
  "	celsius (degC)\n"
  "	ina (mA/cm2)\n"
  "	gna (S/cm2)\n"
  "	minf\n"
  "	mtau (ms)\n"
  "	hinf\n"
  "	htau (ms)\n"
  "	qg ()  : computed q10 for gnabar based on q10g\n"
  "    q10 ()\n"
  "\n"
  "	am (/ms)\n"
  "	bm (/ms)\n"
  "	ah (/ms)\n"
  "	bh (/ms)\n"
  "}\n"
  "\n"
  "STATE {\n"
  "	m h\n"
  "}\n"
  "\n"
  "INITIAL {\n"
  "	qg = q10g^((celsius-22)/10 (degC))\n"
  "    q10 = q10tau^((celsius - 22)/10 (degC)) : if you don't like room temp, it can be changed!\n"
  "	rates(v)\n"
  "	m = minf\n"
  "	h = hinf\n"
  "}\n"
  "\n"
  "BREAKPOINT {\n"
  "	SOLVE state METHOD cnexp\n"
  "	gna = qg*gnabar*(m^3)*h\n"
  "    ina = gna*(v - ena)\n"
  "}\n"
  "\n"
  "DERIVATIVE state {\n"
  "	rates(v)\n"
  "	m' = (minf - m)/mtau\n"
  "	h' = (hinf - h)/htau\n"
  "}\n"
  "\n"
  "PROCEDURE rates(v(mV)) {\n"
  "	am = cam*exp(kam*v)\n"
  "	bm = cbm*exp(kbm*v)\n"
  "\n"
  "	ah = cah*exp(kah*v)\n"
  "	bh = cbh*exp(kbh*v)\n"
  "\n"
  "	minf = am/(am + bm)\n"
  "	mtau = 1/(am + bm)\n"
  "	mtau = mtau/q10\n"
  "	hinf = ah/(ah + bh)\n"
  "	htau = 1/(ah + bh)\n"
  "	htau = htau/q10\n"
  "}\n"
  "\n"
  ;
#endif
