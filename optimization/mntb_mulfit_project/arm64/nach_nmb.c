/* Created by Language version: 7.7.0 */
/* VECTORIZED */
#define NRN_VECTORIZED 1
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
 
#define nrn_init _nrn_init__NaCh_nmb
#define _nrn_initial _nrn_initial__NaCh_nmb
#define nrn_cur _nrn_cur__NaCh_nmb
#define _nrn_current _nrn_current__NaCh_nmb
#define nrn_jacob _nrn_jacob__NaCh_nmb
#define nrn_state _nrn_state__NaCh_nmb
#define _net_receive _net_receive__NaCh_nmb 
#define rates rates__NaCh_nmb 
#define state state__NaCh_nmb 
 
#define _threadargscomma_ _p, _ppvar, _thread, _nt,
#define _threadargsprotocomma_ double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt,
#define _threadargs_ _p, _ppvar, _thread, _nt
#define _threadargsproto_ double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt
 	/*SUPPRESS 761*/
	/*SUPPRESS 762*/
	/*SUPPRESS 763*/
	/*SUPPRESS 765*/
	 extern double *getarg();
 /* Thread safe. No static _p or _ppvar. */
 
#define t _nt->_t
#define dt _nt->_dt
#define gnabar _p[0]
#define gnabar_columnindex 0
#define cam _p[1]
#define cam_columnindex 1
#define kam _p[2]
#define kam_columnindex 2
#define cbm _p[3]
#define cbm_columnindex 3
#define kbm _p[4]
#define kbm_columnindex 4
#define cah _p[5]
#define cah_columnindex 5
#define kah _p[6]
#define kah_columnindex 6
#define cbh _p[7]
#define cbh_columnindex 7
#define kbh _p[8]
#define kbh_columnindex 8
#define ina _p[9]
#define ina_columnindex 9
#define m _p[10]
#define m_columnindex 10
#define h _p[11]
#define h_columnindex 11
#define ena _p[12]
#define ena_columnindex 12
#define gna _p[13]
#define gna_columnindex 13
#define minf _p[14]
#define minf_columnindex 14
#define mtau _p[15]
#define mtau_columnindex 15
#define hinf _p[16]
#define hinf_columnindex 16
#define htau _p[17]
#define htau_columnindex 17
#define am _p[18]
#define am_columnindex 18
#define bm _p[19]
#define bm_columnindex 19
#define ah _p[20]
#define ah_columnindex 20
#define bh _p[21]
#define bh_columnindex 21
#define qg _p[22]
#define qg_columnindex 22
#define q10 _p[23]
#define q10_columnindex 23
#define Dm _p[24]
#define Dm_columnindex 24
#define Dh _p[25]
#define Dh_columnindex 25
#define v _p[26]
#define v_columnindex 26
#define _g _p[27]
#define _g_columnindex 27
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
 static Datum* _extcall_thread;
 static Prop* _extcall_prop;
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
 _extcall_prop = _prop;
 }
 static void _hoc_setdata() {
 Prop *_prop, *hoc_getdata_range(int);
 _prop = hoc_getdata_range(_mechtype);
   _setdata(_prop);
 hoc_retpushx(1.);
}
 /* connect user functions to hoc names */
 static VoidFunc hoc_intfunc[] = {
 "setdata_NaCh_nmb", _hoc_setdata,
 "rates_NaCh_nmb", _hoc_rates,
 0, 0
};
 /* declare global and static user variables */
#define q10g q10g_NaCh_nmb
 double q10g = 2;
#define q10tau q10tau_NaCh_nmb
 double q10tau = 3;
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 0,0,0
};
 static HocParmUnits _hoc_parm_units[] = {
 "gnabar_NaCh_nmb", "S/cm2",
 "cam_NaCh_nmb", "/ms",
 "kam_NaCh_nmb", "/mV",
 "cbm_NaCh_nmb", "/ms",
 "kbm_NaCh_nmb", "/mV",
 "cah_NaCh_nmb", "/ms",
 "kah_NaCh_nmb", "/mV",
 "cbh_NaCh_nmb", "/ms",
 "kbh_NaCh_nmb", "/mV",
 "ina_NaCh_nmb", "mA/cm2",
 0,0
};
 static double delta_t = 0.01;
 static double h0 = 0;
 static double m0 = 0;
 /* connect global user variables to hoc */
 static DoubScal hoc_scdoub[] = {
 "q10tau_NaCh_nmb", &q10tau_NaCh_nmb,
 "q10g_NaCh_nmb", &q10g_NaCh_nmb,
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
"NaCh_nmb",
 "gnabar_NaCh_nmb",
 "cam_NaCh_nmb",
 "kam_NaCh_nmb",
 "cbm_NaCh_nmb",
 "kbm_NaCh_nmb",
 "cah_NaCh_nmb",
 "kah_NaCh_nmb",
 "cbh_NaCh_nmb",
 "kbh_NaCh_nmb",
 0,
 "ina_NaCh_nmb",
 0,
 "m_NaCh_nmb",
 "h_NaCh_nmb",
 0,
 0};
 static Symbol* _na_sym;
 
extern Prop* need_memb(Symbol*);

static void nrn_alloc(Prop* _prop) {
	Prop *prop_ion;
	double *_p; Datum *_ppvar;
 	_p = nrn_prop_data_alloc(_mechtype, 28, _prop);
 	/*initialize range parameters*/
 	gnabar = 0.05;
 	cam = 76.4;
 	kam = 0.037;
 	cbm = 6.93085;
 	kbm = -0.043;
 	cah = 0.000533;
 	kah = -0.0909;
 	cbh = 0.787;
 	kbh = 0.0691;
 	_prop->param = _p;
 	_prop->param_size = 28;
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

 void _nach_nmb_reg() {
	int _vectorized = 1;
  _initlists();
 	ion_reg("na", -10000.);
 	_na_sym = hoc_lookup("na_ion");
 	register_mech(_mechanism, nrn_alloc,nrn_cur, nrn_jacob, nrn_state, nrn_init, hoc_nrnpointerindex, 1);
 _mechtype = nrn_get_mechtype(_mechanism[1]);
     _nrn_setdata_reg(_mechtype, _setdata);
     _nrn_thread_reg(_mechtype, 2, _update_ion_pointer);
 #if NMODL_TEXT
  hoc_reg_nmodl_text(_mechtype, nmodl_file_text);
  hoc_reg_nmodl_filename(_mechtype, nmodl_filename);
#endif
  hoc_register_prop_size(_mechtype, 28, 4);
  hoc_register_dparam_semantics(_mechtype, 0, "na_ion");
  hoc_register_dparam_semantics(_mechtype, 1, "na_ion");
  hoc_register_dparam_semantics(_mechtype, 2, "na_ion");
  hoc_register_dparam_semantics(_mechtype, 3, "cvodeieq");
 	hoc_register_cvode(_mechtype, _ode_count, _ode_map, _ode_spec, _ode_matsol);
 	hoc_register_tolerance(_mechtype, _hoc_state_tol, &_atollist);
 	hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 NaCh_nmb /Users/nikollas/Library/CloudStorage/OneDrive-UniversityofSouthFlorida/MNTB_neuron/optimization/mntb_fit_project/mod/nach_nmb.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
static int _reset;
static char *modelname = "";

static int error;
static int _ninits = 0;
static int _match_recurse=1;
static void _modl_cleanup(){ _match_recurse=1;}
static int rates(_threadargsprotocomma_ double);
 
static int _ode_spec1(_threadargsproto_);
/*static int _ode_matsol1(_threadargsproto_);*/
 static int _slist1[2], _dlist1[2];
 static int state(_threadargsproto_);
 
/*CVODE*/
 static int _ode_spec1 (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {int _reset = 0; {
   rates ( _threadargscomma_ v ) ;
   Dm = ( minf - m ) / mtau ;
   Dh = ( hinf - h ) / htau ;
   }
 return _reset;
}
 static int _ode_matsol1 (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
 rates ( _threadargscomma_ v ) ;
 Dm = Dm  / (1. - dt*( ( ( ( - 1.0 ) ) ) / mtau )) ;
 Dh = Dh  / (1. - dt*( ( ( ( - 1.0 ) ) ) / htau )) ;
  return 0;
}
 /*END CVODE*/
 static int state (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) { {
   rates ( _threadargscomma_ v ) ;
    m = m + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / mtau)))*(- ( ( ( minf ) ) / mtau ) / ( ( ( ( - 1.0 ) ) ) / mtau ) - m) ;
    h = h + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / htau)))*(- ( ( ( hinf ) ) / htau ) / ( ( ( ( - 1.0 ) ) ) / htau ) - h) ;
   }
  return 0;
}
 
static int  rates ( _threadargsprotocomma_ double _lv ) {
   double _lam , _lbm , _lah , _lbh ;
 _lam = cam * exp ( kam * _lv ) ;
   _lbm = cbm * exp ( kbm * _lv ) ;
   _lah = cah * exp ( kah * _lv ) ;
   _lbh = cbh * exp ( kbh * _lv ) ;
   mtau = 1.0 / ( _lam + _lbm ) ;
   mtau = mtau / q10 ;
   minf = _lam / ( _lam + _lbm ) ;
   hinf = _lah / ( _lah + _lbh ) ;
   htau = 1.0 / ( _lah + _lbh ) ;
   htau = htau / q10 ;
    return 0; }
 
static void _hoc_rates(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r = 1.;
 rates ( _p, _ppvar, _thread, _nt, *getarg(1) );
 hoc_retpushx(_r);
}
 
static int _ode_count(int _type){ return 2;}
 
static void _ode_spec(NrnThread* _nt, _Memb_list* _ml, int _type) {
   double* _p; Datum* _ppvar; Datum* _thread;
   Node* _nd; double _v; int _iml, _cntml;
  _cntml = _ml->_nodecount;
  _thread = _ml->_thread;
  for (_iml = 0; _iml < _cntml; ++_iml) {
    _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
    _nd = _ml->_nodelist[_iml];
    v = NODEV(_nd);
  ena = _ion_ena;
     _ode_spec1 (_p, _ppvar, _thread, _nt);
  }}
 
static void _ode_map(int _ieq, double** _pv, double** _pvdot, double* _pp, Datum* _ppd, double* _atol, int _type) { 
	double* _p; Datum* _ppvar;
 	int _i; _p = _pp; _ppvar = _ppd;
	_cvode_ieq = _ieq;
	for (_i=0; _i < 2; ++_i) {
		_pv[_i] = _pp + _slist1[_i];  _pvdot[_i] = _pp + _dlist1[_i];
		_cvode_abstol(_atollist, _atol, _i);
	}
 }
 
static void _ode_matsol_instance1(_threadargsproto_) {
 _ode_matsol1 (_p, _ppvar, _thread, _nt);
 }
 
static void _ode_matsol(NrnThread* _nt, _Memb_list* _ml, int _type) {
   double* _p; Datum* _ppvar; Datum* _thread;
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

static void initmodel(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
  int _i; double _save;{
  h = h0;
  m = m0;
 {
   qg = pow( q10g , ( ( celsius - 22.0 ) / 10.0 ) ) ;
   q10 = pow( q10tau , ( ( celsius - 22.0 ) / 10.0 ) ) ;
   rates ( _threadargscomma_ v ) ;
   m = minf ;
   h = hinf ;
   }
 
}
}

static void nrn_init(NrnThread* _nt, _Memb_list* _ml, int _type){
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; double _v; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
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
 initmodel(_p, _ppvar, _thread, _nt);
 }
}

static double _nrn_current(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt, double _v){double _current=0.;v=_v;{ {
   gna = qg * gnabar * ( pow( m , 3.0 ) ) * h ;
   ina = gna * ( v - ena ) ;
   }
 _current += ina;

} return _current;
}

static void nrn_cur(NrnThread* _nt, _Memb_list* _ml, int _type) {
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; int* _ni; double _rhs, _v; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
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
 _g = _nrn_current(_p, _ppvar, _thread, _nt, _v + .001);
 	{ double _dina;
  _dina = ina;
 _rhs = _nrn_current(_p, _ppvar, _thread, _nt, _v);
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
 
}
 
}

static void nrn_jacob(NrnThread* _nt, _Memb_list* _ml, int _type) {
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
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
 
}
 
}

static void nrn_state(NrnThread* _nt, _Memb_list* _ml, int _type) {
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; double _v = 0.0; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
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
 {   state(_p, _ppvar, _thread, _nt);
  } }}

}

static void terminal(){}

static void _initlists(){
 double _x; double* _p = &_x;
 int _i; static int _first = 1;
  if (!_first) return;
 _slist1[0] = m_columnindex;  _dlist1[0] = Dm_columnindex;
 _slist1[1] = h_columnindex;  _dlist1[1] = Dh_columnindex;
_first = 0;
}

#if defined(__cplusplus)
} /* extern "C" */
#endif

#if NMODL_TEXT
static const char* nmodl_filename = "/Users/nikollas/Library/CloudStorage/OneDrive-UniversityofSouthFlorida/MNTB_neuron/optimization/mntb_fit_project/mod/nach_nmb.mod";
static const char* nmodl_file_text = 
  "NEURON {\n"
  "    SUFFIX NaCh_nmb\n"
  "    USEION na READ ena WRITE ina\n"
  "    RANGE gnabar, ina\n"
  "    RANGE cam, kam, cbm, kbm\n"
  "    RANGE cah, kah, cbh, kbh\n"
  "}\n"
  "\n"
  "UNITS {\n"
  "    (S) = (siemens)\n"
  "    (mV) = (millivolt)\n"
  "    (mA) = (milliamp)\n"
  "}\n"
  "\n"
  "PARAMETER {\n"
  "    v (mV)\n"
  "    gnabar = 0.05 (S/cm2)\n"
  "    ena (mV)\n"
  "	q10tau = 3\n"
  "	q10g = 2\n"
  "\n"
  "    cam = 76.4 (/ms)\n"
  "    kam = 0.037 (/mV)\n"
  "    cbm = 6.930852 (/ms)\n"
  "    kbm = -0.043 (/mV)\n"
  "\n"
  "    cah = 0.000533 (/ms)\n"
  "    kah = -0.0909 (/mV)\n"
  "    cbh = 0.787 (/ms)\n"
  "    kbh = 0.0691 (/mV)\n"
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
  "\n"
  "	am (/ms)\n"
  "	bm (/ms)\n"
  "	ah (/ms)\n"
  "	bh (/ms)\n"
  "\n"
  "	qg ()  : computed q10 for gnabar based on q10g\n"
  "    q10 ()\n"
  "}\n"
  "\n"
  "STATE {\n"
  "    m h\n"
  "}\n"
  "\n"
  "INITIAL {\n"
  "    qg = q10g^((celsius-22)/10 (degC))\n"
  "    q10 = q10tau^((celsius - 22)/10 (degC)) : if you don't like room temp, it can be changed!\n"
  "	rates(v)\n"
  "    m = minf\n"
  "    h = hinf\n"
  "}\n"
  "\n"
  "BREAKPOINT {\n"
  "    SOLVE state METHOD cnexp\n"
  "    gna = qg*gnabar*(m^3)*h\n"
  "    ina = gna*(v - ena)\n"
  "}\n"
  "\n"
  "DERIVATIVE state {\n"
  "    rates(v)\n"
  "    m' = (minf - m)/mtau\n"
  "    h' = (hinf - h)/htau\n"
  "}\n"
  "\n"
  "PROCEDURE rates(v (mV)) {\n"
  "    LOCAL am, bm, ah, bh\n"
  "\n"
  "    am = cam * exp(kam * v)\n"
  "    bm = cbm * exp(kbm * v)\n"
  "    ah = cah * exp(kah * v)\n"
  "    bh = cbh * exp(kbh * v)\n"
  "\n"
  "    mtau = 1 / (am + bm)\n"
  "	mtau = mtau/q10\n"
  "    minf = am / (am + bm)\n"
  "	hinf = ah / (ah + bh)\n"
  "    htau = 1 / (ah + bh)\n"
  "	htau = htau/q10\n"
  "\n"
  "}\n"
  ;
#endif
