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
 
#define nrn_init _nrn_init__hcno
#define _nrn_initial _nrn_initial__hcno
#define nrn_cur _nrn_cur__hcno
#define _nrn_current _nrn_current__hcno
#define nrn_jacob _nrn_jacob__hcno
#define nrn_state _nrn_state__hcno
#define _net_receive _net_receive__hcno 
#define states states__hcno 
#define trates trates__hcno 
 
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
#define i _p[1]
#define i_columnindex 1
#define h1 _p[2]
#define h1_columnindex 2
#define h2 _p[3]
#define h2_columnindex 3
#define thegna _p[4]
#define thegna_columnindex 4
#define Dh1 _p[5]
#define Dh1_columnindex 5
#define Dh2 _p[6]
#define Dh2_columnindex 6
#define _g _p[7]
#define _g_columnindex 7
 
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
 static void _hoc_alp2(void);
 static void _hoc_alp1(void);
 static void _hoc_bet2(void);
 static void _hoc_bet1(void);
 static void _hoc_trates(void);
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
 "setdata_hcno", _hoc_setdata,
 "alp2_hcno", _hoc_alp2,
 "alp1_hcno", _hoc_alp1,
 "bet2_hcno", _hoc_bet2,
 "bet1_hcno", _hoc_bet1,
 "trates_hcno", _hoc_trates,
 0, 0
};
#define alp2 alp2_hcno
#define alp1 alp1_hcno
#define bet2 bet2_hcno
#define bet1 bet1_hcno
 extern double alp2( double );
 extern double alp1( double );
 extern double bet2( double );
 extern double bet1( double );
 /* declare global and static user variables */
#define a02 a02_hcno
 double a02 = 0.0029;
#define a01 a01_hcno
 double a01 = 0.008;
#define eh eh_hcno
 double eh = 0;
#define frac frac_hcno
 double frac = 0;
#define gm2 gm2_hcno
 double gm2 = 0.6;
#define gm1 gm1_hcno
 double gm1 = 0.3;
#define hinf hinf_hcno
 double hinf = 0;
#define q10 q10_hcno
 double q10 = 4.5;
#define qinf qinf_hcno
 double qinf = 7;
#define thinf thinf_hcno
 double thinf = -66;
#define tau2 tau2_hcno
 double tau2 = 0;
#define tau1 tau1_hcno
 double tau1 = 0;
#define vhalf2 vhalf2_hcno
 double vhalf2 = -84;
#define vhalf1 vhalf1_hcno
 double vhalf1 = -50;
#define zeta2 zeta2_hcno
 double zeta2 = 3;
#define zeta1 zeta1_hcno
 double zeta1 = 3;
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 0,0,0
};
 static HocParmUnits _hoc_parm_units[] = {
 "vhalf1_hcno", "mV",
 "vhalf2_hcno", "mV",
 "gm1_hcno", "mV",
 "gm2_hcno", "mV",
 "zeta1_hcno", "/ms",
 "zeta2_hcno", "/ms",
 "thinf_hcno", "mV",
 "qinf_hcno", "mV",
 "eh_hcno", "mV",
 "gbar_hcno", "mho/cm2",
 "i_hcno", "mA/cm2",
 0,0
};
 static double delta_t = 0.01;
 static double h20 = 0;
 static double h10 = 0;
 static double v = 0;
 /* connect global user variables to hoc */
 static DoubScal hoc_scdoub[] = {
 "vhalf1_hcno", &vhalf1_hcno,
 "vhalf2_hcno", &vhalf2_hcno,
 "gm1_hcno", &gm1_hcno,
 "gm2_hcno", &gm2_hcno,
 "zeta1_hcno", &zeta1_hcno,
 "zeta2_hcno", &zeta2_hcno,
 "a01_hcno", &a01_hcno,
 "a02_hcno", &a02_hcno,
 "frac_hcno", &frac_hcno,
 "thinf_hcno", &thinf_hcno,
 "qinf_hcno", &qinf_hcno,
 "q10_hcno", &q10_hcno,
 "eh_hcno", &eh_hcno,
 "hinf_hcno", &hinf_hcno,
 "tau1_hcno", &tau1_hcno,
 "tau2_hcno", &tau2_hcno,
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
 
#define _cvode_ieq _ppvar[0]._i
 static void _ode_matsol_instance1(_threadargsproto_);
 /* connect range variables in _p that hoc is supposed to know about */
 static const char *_mechanism[] = {
 "7.7.0",
"hcno",
 "gbar_hcno",
 0,
 "i_hcno",
 0,
 "h1_hcno",
 "h2_hcno",
 0,
 0};
 
extern Prop* need_memb(Symbol*);

static void nrn_alloc(Prop* _prop) {
	Prop *prop_ion;
	double *_p; Datum *_ppvar;
 	_p = nrn_prop_data_alloc(_mechtype, 8, _prop);
 	/*initialize range parameters*/
 	gbar = 0.0005;
 	_prop->param = _p;
 	_prop->param_size = 8;
 	_ppvar = nrn_prop_datum_alloc(_mechtype, 1, _prop);
 	_prop->dparam = _ppvar;
 	/*connect ionic variables to this model*/
 
}
 static void _initlists();
  /* some states have an absolute tolerance */
 static Symbol** _atollist;
 static HocStateTolerance _hoc_state_tol[] = {
 0,0
};
 extern Symbol* hoc_lookup(const char*);
extern void _nrn_thread_reg(int, int, void(*)(Datum*));
extern void _nrn_thread_table_reg(int, void(*)(double*, Datum*, Datum*, NrnThread*, int));
extern void hoc_register_tolerance(int, HocStateTolerance*, Symbol***);
extern void _cvode_abstol( Symbol**, double*, int);

 void _hcno_reg() {
	int _vectorized = 0;
  _initlists();
 	register_mech(_mechanism, nrn_alloc,nrn_cur, nrn_jacob, nrn_state, nrn_init, hoc_nrnpointerindex, 0);
 _mechtype = nrn_get_mechtype(_mechanism[1]);
     _nrn_setdata_reg(_mechtype, _setdata);
 #if NMODL_TEXT
  hoc_reg_nmodl_text(_mechtype, nmodl_file_text);
  hoc_reg_nmodl_filename(_mechtype, nmodl_filename);
#endif
  hoc_register_prop_size(_mechtype, 8, 1);
  hoc_register_dparam_semantics(_mechtype, 0, "cvodeieq");
 	hoc_register_cvode(_mechtype, _ode_count, _ode_map, _ode_spec, _ode_matsol);
 	hoc_register_tolerance(_mechtype, _hoc_state_tol, &_atollist);
 	hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 hcno hcno.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
static int _reset;
static char *modelname = "h current for Octopus cells of Cochlear Nucleus";

static int error;
static int _ninits = 0;
static int _match_recurse=1;
static void _modl_cleanup(){ _match_recurse=1;}
static int trates(double);
 static int _deriv1_advance = 0;
 
static int _ode_spec1(_threadargsproto_);
/*static int _ode_matsol1(_threadargsproto_);*/
 static int _slist2[2]; static double _dlist2[2];
 static double _savstate1[2], *_temp1 = _savstate1;
 static int _slist1[2], _dlist1[2];
 static int states(_threadargsproto_);
 
/*CVODE*/
 static int _ode_spec1 () {_reset=0;
 {
   trates ( _threadargscomma_ v ) ;
   Dh1 = ( hinf - h1 ) / tau1 ;
   Dh2 = ( hinf - h2 ) / tau2 ;
   }
 return _reset;
}
 static int _ode_matsol1 () {
 trates ( _threadargscomma_ v ) ;
 Dh1 = Dh1  / (1. - dt*( ( ( ( - 1.0 ) ) ) / tau1 )) ;
 Dh2 = Dh2  / (1. - dt*( ( ( ( - 1.0 ) ) ) / tau2 )) ;
  return 0;
}
 /*END CVODE*/
 
static int states () {_reset=0;
 { static int _recurse = 0;
 int _counte = -1;
 if (!_recurse) {
 _recurse = 1;
 {int _id; for(_id=0; _id < 2; _id++) { _savstate1[_id] = _p[_slist1[_id]];}}
 error = newton(2,_slist2, _p, states, _dlist2);
 _recurse = 0; if(error) {abort_run(error);}}
 {
   trates ( _threadargscomma_ v ) ;
   Dh1 = ( hinf - h1 ) / tau1 ;
   Dh2 = ( hinf - h2 ) / tau2 ;
   {int _id; for(_id=0; _id < 2; _id++) {
if (_deriv1_advance) {
 _dlist2[++_counte] = _p[_dlist1[_id]] - (_p[_slist1[_id]] - _savstate1[_id])/dt;
 }else{
_dlist2[++_counte] = _p[_slist1[_id]] - _savstate1[_id];}}}
 } }
 return _reset;}
 
static int  trates (  double _lv ) {
   double _lqt ;
 _lqt = pow( q10 , ( ( celsius - 33.0 ) / 10.0 ) ) ;
   tau1 = bet1 ( _threadargscomma_ _lv ) / ( _lqt * a01 * ( 1.0 + alp1 ( _threadargscomma_ _lv ) ) ) ;
   tau2 = bet2 ( _threadargscomma_ _lv ) / ( _lqt * a02 * ( 1.0 + alp2 ( _threadargscomma_ _lv ) ) ) ;
   hinf = 1.0 / ( 1.0 + exp ( ( _lv - thinf ) / qinf ) ) ;
    return 0; }
 
static void _hoc_trates(void) {
  double _r;
   _r = 1.;
 trates (  *getarg(1) );
 hoc_retpushx(_r);
}
 
double alp1 (  double _lv ) {
   double _lalp1;
 _lalp1 = exp ( 1.e-3 * zeta1 * ( _lv - vhalf1 ) * 9.648e4 / ( 8.315 * ( 273.16 + celsius ) ) ) ;
   
return _lalp1;
 }
 
static void _hoc_alp1(void) {
  double _r;
   _r =  alp1 (  *getarg(1) );
 hoc_retpushx(_r);
}
 
double bet1 (  double _lv ) {
   double _lbet1;
 _lbet1 = exp ( 1.e-3 * zeta1 * gm1 * ( _lv - vhalf1 ) * 9.648e4 / ( 8.315 * ( 273.16 + celsius ) ) ) ;
   
return _lbet1;
 }
 
static void _hoc_bet1(void) {
  double _r;
   _r =  bet1 (  *getarg(1) );
 hoc_retpushx(_r);
}
 
double alp2 (  double _lv ) {
   double _lalp2;
 _lalp2 = exp ( 1.e-3 * zeta2 * ( _lv - vhalf2 ) * 9.648e4 / ( 8.315 * ( 273.16 + celsius ) ) ) ;
   
return _lalp2;
 }
 
static void _hoc_alp2(void) {
  double _r;
   _r =  alp2 (  *getarg(1) );
 hoc_retpushx(_r);
}
 
double bet2 (  double _lv ) {
   double _lbet2;
 _lbet2 = exp ( 1.e-3 * zeta2 * gm2 * ( _lv - vhalf2 ) * 9.648e4 / ( 8.315 * ( 273.16 + celsius ) ) ) ;
   
return _lbet2;
 }
 
static void _hoc_bet2(void) {
  double _r;
   _r =  bet2 (  *getarg(1) );
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
 _ode_matsol_instance1(_threadargs_);
 }}

static void initmodel() {
  int _i; double _save;_ninits++;
 _save = t;
 t = 0.0;
{
  h2 = h20;
  h1 = h10;
 {
   trates ( _threadargscomma_ v ) ;
   h1 = hinf ;
   h2 = hinf ;
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
 initmodel();
}}

static double _nrn_current(double _v){double _current=0.;v=_v;{ {
   thegna = gbar * ( h1 * frac + h2 * ( 1.0 - frac ) ) ;
   i = thegna * ( v - eh ) ;
   }
 _current += i;

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
 _g = _nrn_current(_v + .001);
 	{ _rhs = _nrn_current(_v);
 	}
 _g = (_g - _rhs)/.001;
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
double _dtsav = dt;
if (secondorder) { dt *= 0.5; }
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
 { error = _deriv1_advance = 1;
 derivimplicit(_ninits, 2, _slist1, _dlist1, _p, &t, dt, states, &_temp1);
_deriv1_advance = 0;
 if(error){fprintf(stderr,"at line 54 in file hcno.mod:\n        SOLVE states METHOD derivimplicit\n"); nrn_complain(_p); abort_run(error);}
    if (secondorder) {
    int _i;
    for (_i = 0; _i < 2; ++_i) {
      _p[_slist1[_i]] += dt*_p[_dlist1[_i]];
    }}
 }}}
 dt = _dtsav;
}

static void terminal(){}

static void _initlists() {
 int _i; static int _first = 1;
  if (!_first) return;
 _slist1[0] = h1_columnindex;  _dlist1[0] = Dh1_columnindex;
 _slist1[1] = h2_columnindex;  _dlist1[1] = Dh2_columnindex;
 _slist2[0] = h2_columnindex;
 _slist2[1] = h1_columnindex;
_first = 0;
}

#if NMODL_TEXT
static const char* nmodl_filename = "hcno.mod";
static const char* nmodl_file_text = 
  "TITLE h current for Octopus cells of Cochlear Nucleus\n"
  ": From Bal and Oertel (2000)\n"
  ": M.Migliore Oct. 2001\n"
  "\n"
  "NEURON {\n"
  "	SUFFIX hcno\n"
  "	NONSPECIFIC_CURRENT i\n"
  "	RANGE  gbar\n"
  "	GLOBAL hinf, tau1,tau2\n"
  "}\n"
  "\n"
  "PARAMETER {\n"
  "	gbar = 0.0005   	(mho/cm2)	\n"
  "								\n"
  "	vhalf1  = -50	(mV)		: v 1/2 for forward\n"
  "	vhalf2  = -84 	(mV)		: v 1/2 for backward	\n"
  "	gm1   = 0.3	(mV)	        : slope for forward\n"
  "	gm2   = 0.6      (mV)		: slope for backward\n"
  "	zeta1   = 3 	(/ms)		\n"
  "	zeta2   = 3 	(/ms)		\n"
  "	a01 = 0.008 \n"
  "	a02 = 0.0029\n"
  "	frac=0.0\n"
  "\n"
  "\n"
  "	thinf  = -66 	(mV)		: inact inf slope	\n"
  "	qinf  = 7 	(mV)		: inact inf slope \n"
  "\n"
  "	q10=4.5				: from Magee (1998)\n"
  "\n"
  "	eh		(mV)            : must be explicitly def. in hoc\n"
  "	celsius\n"
  "	v 		(mV)\n"
  "}\n"
  "\n"
  "\n"
  "UNITS {\n"
  "	(mA) = (milliamp)\n"
  "	(mV) = (millivolt)\n"
  "	(pS) = (picosiemens)\n"
  "	(um) = (micron)\n"
  "} \n"
  "\n"
  "ASSIGNED {\n"
  "	i 		(mA/cm2)\n"
  "	thegna		(mho/cm2)\n"
  "	hinf tau1 tau2 \n"
  "}\n"
  " \n"
  "\n"
  "STATE { h1 h2 }\n"
  "\n"
  "BREAKPOINT {\n"
  "        SOLVE states METHOD derivimplicit\n"
  "        thegna = gbar*(h1*frac + h2*(1-frac))\n"
  "	i = thegna * (v - eh)\n"
  "} \n"
  "\n"
  "INITIAL {\n"
  "	trates(v)\n"
  "	h1=hinf\n"
  "	h2=hinf\n"
  "}\n"
  "\n"
  "DERIVATIVE states {   \n"
  "        trates(v)      \n"
  "		h1' = (hinf - h1)/tau1\n"
  "		h2' = (hinf - h2)/tau2\n"
  "}\n"
  "\n"
  "PROCEDURE trates(v) {  \n"
  "        LOCAL  qt\n"
  "        qt=q10^((celsius-33)/10)\n"
  "\n"
  "        tau1 = bet1(v)/(qt*a01*(1+alp1(v)))\n"
  "        tau2 = bet2(v)/(qt*a02*(1+alp2(v)))\n"
  "\n"
  "	hinf = 1/(1+exp((v-thinf)/qinf))\n"
  "}\n"
  "\n"
  "FUNCTION alp1(v(mV)) {\n"
  "  alp1 = exp(1.e-3*zeta1*(v-vhalf1)*9.648e4/(8.315*(273.16+celsius))) \n"
  "}\n"
  "\n"
  "FUNCTION bet1(v(mV)) {\n"
  "  bet1 = exp(1.e-3*zeta1*gm1*(v-vhalf1)*9.648e4/(8.315*(273.16+celsius))) \n"
  "}\n"
  "\n"
  "FUNCTION alp2(v(mV)) {\n"
  "  alp2 = exp(1.e-3*zeta2*(v-vhalf2)*9.648e4/(8.315*(273.16+celsius))) \n"
  "}\n"
  "\n"
  "FUNCTION bet2(v(mV)) {\n"
  "  bet2 = exp(1.e-3*zeta2*gm2*(v-vhalf2)*9.648e4/(8.315*(273.16+celsius))) \n"
  "}\n"
  ;
#endif
