#ifndef MHD_BASE_FUNCTOR_2D_H_
#define MHD_BASE_FUNCTOR_2D_H_

#include "kokkos_shared.h"

#include "HydroParams.h"
#include "HydroState.h"

namespace ppkMHD { namespace muscl {

/**
 * Base class to derive actual kokkos functor.
 * params is passed by copy.
 */
class MHDBaseFunctor2D
{

public:

  using HydroState = MHDState;
  using DataArray  = DataArray2d;

  MHDBaseFunctor2D(HydroParams params) : params(params) {};
  virtual ~MHDBaseFunctor2D() {};

  HydroParams params;
  const int nbvar = params.nbvar;

  // utility routines used in various computational kernels

  KOKKOS_INLINE_FUNCTION
  void swapValues(real_t *a, real_t *b) const
  {
    
    real_t tmp = *a;
    
    *a = *b;
    *b = tmp;
    
  } // swapValues
  
  /**
   * max value out of 4
   */
  KOKKOS_INLINE_FUNCTION
  real_t FMAX4(real_t a0, real_t a1, real_t a2, real_t a3) const 
  {
    real_t returnVal = a0;
    returnVal = ( a1 > returnVal) ? a1 : returnVal;
    returnVal = ( a2 > returnVal) ? a2 : returnVal;
    returnVal = ( a3 > returnVal) ? a3 : returnVal;
    
    return returnVal;
  } // FMAX4
  
  /**
   * min value out of 4
   */
  KOKKOS_INLINE_FUNCTION
  real_t FMIN4(real_t a0, real_t a1, real_t a2, real_t a3) const 
  {
    real_t returnVal = a0;
    returnVal = ( a1 < returnVal) ? a1 : returnVal;
    returnVal = ( a2 < returnVal) ? a2 : returnVal;
    returnVal = ( a3 < returnVal) ? a3 : returnVal;
    
    return returnVal;
  } // FMIN4
  
  /**
   * max value out of 5
   */
  KOKKOS_INLINE_FUNCTION
  real_t FMAX5(real_t a0, real_t a1, real_t a2, real_t a3, real_t a4) const
  {
    real_t returnVal = a0;
    returnVal = ( a1 > returnVal) ? a1 : returnVal;
    returnVal = ( a2 > returnVal) ? a2 : returnVal;
    returnVal = ( a3 > returnVal) ? a3 : returnVal;
    returnVal = ( a4 > returnVal) ? a4 : returnVal;
    
    return returnVal;
  } // FMAX5

  /**
   * Copy data(index) into q.
   */
  KOKKOS_INLINE_FUNCTION
  void get_state(DataArray data, int i, int j, MHDState& q) const
  {

    q.d  = data(i,j, ID);
    q.p  = data(i,j, IP);
    q.u  = data(i,j, IU);
    q.v  = data(i,j, IV);
    q.w  = data(i,j, IW);
    q.bx = data(i,j, IBX);
    q.by = data(i,j, IBY);
    q.bz = data(i,j, IBZ);
    
  } // get_state

  /**
   * Copy q into data(i,j).
   */
  KOKKOS_INLINE_FUNCTION
  void set_state(DataArray data, int i, int j, const MHDState& q) const
  {

    data(i,j, ID)  = q.d;
    data(i,j, IP)  = q.p;
    data(i,j, IU)  = q.u;
    data(i,j, IV)  = q.v;
    data(i,j, IW)  = q.w;
    data(i,j, IBX) = q.bx;
    data(i,j, IBY) = q.by;
    data(i,j, IBZ) = q.bz;
    
  } // set_state

  /**
   *
   */
  KOKKOS_INLINE_FUNCTION
  void get_magField(const DataArray& data, int i, int j, BField& b) const
  {

    b.bx = data(i,j, IBX);
    b.by = data(i,j, IBY);
    b.bz = data(i,j, IBZ);
    
  } // get_magField
  
  /**
   * Equation of state:
   * compute pressure p and speed of sound c, from density rho and
   * internal energy eint using the "calorically perfect gas" equation
   * of state : \f$ eint=\frac{p}{\rho (\gamma-1)} \f$
   * Recall that \f$ \gamma \f$ is equal to the ratio of specific heats
   *  \f$ \left[ c_p/c_v \right] \f$.
   * 
   * @param[in]  rho  density
   * @param[in]  eint internal energy
   * @param[out] p    pressure
   * @param[out] c    speed of sound
   */
  KOKKOS_INLINE_FUNCTION
  void eos(real_t rho,
	   real_t eint,
	   real_t* p,
	   real_t* c) const
  {
    real_t gamma0 = params.settings.gamma0;
    real_t smallp = params.settings.smallp;
    
    *p = fmax((gamma0 - ONE_F) * rho * eint, rho * smallp);
    *c = sqrt(gamma0 * (*p) / rho);
    
  } // eos
  
  /**
   * Convert conservative variables (rho, rho*u, rho*v, rho*w, e, bx, by, bz) 
   * to primitive variables (rho,u,v,w,p,bx,by,bz
   *)
   * @param[in]  u  conservative variables array
   * @param[in]  magFieldNeighbors face-centered magnetic fields in neighboring cells.
   * @param[out] c  local speed of sound
   * @param[out] q  primitive    variables array (allocated in calling routine, size is constant NBVAR)
   */
  KOKKOS_INLINE_FUNCTION
  void constoprim_mhd(const MHDState& u,
		      const real_t magFieldNeighbors[3],
		      real_t &c,
		      MHDState &q) const
  {
    real_t smallr = params.settings.smallr;

    // compute density
    q.d = fmax(u.d, smallr);

    // compute velocities
    q.u = u.u / q.d;
    q.v = u.v / q.d;
    q.w = u.w / q.d;

    // compute cell-centered magnetic field
    q.bx = 0.5 * ( u.bx + magFieldNeighbors[0] );
    q.by = 0.5 * ( u.by + magFieldNeighbors[1] );
    q.bz = 0.5 * ( u.bz + magFieldNeighbors[2] );

    // compute specific kinetic energy and magnetic energy
    real_t eken = 0.5 * (q.u *q.u  + q.v *q.v  + q.w *q.w );
    real_t emag = 0.5 * (q.bx*q.bx + q.by*q.by + q.bz*q.bz);

    // compute pressure

    if (params.settings.cIso > 0) { // isothermal
      
      q.p = q.d * (params.settings.cIso) * (params.settings.cIso);
      c     =  params.settings.cIso;
      
    } else {
      
      real_t eint = (u.p - emag) / q.d - eken;

      q.p = fmax((params.settings.gamma0-1.0) * q.d * eint,
		 q.d * params.settings.smallp);
  
      // if (q.p < 0) {
      // 	printf("MHD pressure neg !!!\n");
      // }

      // compute speed of sound (should be removed as it is useless, hydro
      // legacy)
      c = sqrt(params.settings.gamma0 * q.p / q.d);
    }
 
  } // constoprim_mhd

    /**
     * Compute primitive variables slopes (dqX,dqY) for one component from q and its neighbors.
     * This routine is only used in the 2D UNSPLIT integration and slope_type = 0,1 and 2.
     * 
     * Only slope_type 1 and 2 are supported.
     *
     * \param[in]  q       : current primitive variable
     * \param[in]  qPlusX  : value in the next neighbor cell along XDIR
     * \param[in]  qMinusX : value in the previous neighbor cell along XDIR
     * \param[in]  qPlusY  : value in the next neighbor cell along YDIR
     * \param[in]  qMinusY : value in the previous neighbor cell along YDIR
     * \param[out] dqX     : reference to an array returning the X slopes
     * \param[out] dqY     : reference to an array returning the Y slopes
     *
     */
  KOKKOS_INLINE_FUNCTION
  void slope_unsplit_hydro_2d_scalar(real_t q, 
				     real_t qPlusX,
				     real_t qMinusX,
				     real_t qPlusY,
				     real_t qMinusY,
				     real_t *dqX,
				     real_t *dqY) const
  {
    real_t slope_type = params.settings.slope_type;

    real_t dlft, drgt, dcen, dsgn, slop, dlim;

    // slopes in first coordinate direction
    dlft = slope_type*(q      - qMinusX);
    drgt = slope_type*(qPlusX - q      );
    dcen = 0.5 * (qPlusX - qMinusX);
    dsgn = (dcen >= ZERO_F) ? ONE_F : -ONE_F;
    slop = fmin( FABS(dlft), FABS(drgt) );
    dlim = slop;
    if ( (dlft*drgt) <= ZERO_F )
      dlim = ZERO_F;
    *dqX = dsgn * fmin( dlim, FABS(dcen) );
  
    // slopes in second coordinate direction
    dlft = slope_type*(q      - qMinusY);
    drgt = slope_type*(qPlusY - q      );
    dcen = 0.5 * (qPlusY - qMinusY);
    dsgn = (dcen >= ZERO_F) ? ONE_F : -ONE_F;
    slop = fmin( FABS(dlft), FABS(drgt) );
    dlim = slop;
    if ( (dlft*drgt) <= ZERO_F )
      dlim = ZERO_F;
    *dqY = dsgn * fmin( dlim, FABS(dcen) );

  } // slope_unsplit_hydro_2d_scalar

  /**
   * Compute primitive variables slope (vector dq) from q and its neighbors.
   * This routine is only used in the 2D UNSPLIT integration and slope_type = 0,1,2 and 3.
   * 
   * Note that slope_type is a global variable, located in symbol memory when 
   * using the GPU version.
   *
   * Loosely adapted from RAMSES/hydro/umuscl.f90: subroutine uslope
   * Interface is changed to become cellwise.
   * Only slope_type 1 and 2 are supported.
   *
   * \param[in]  qNb     : array to primitive variable vector state in the neighborhood
   * \param[out] dq      : reference to an array returning the X and Y slopes
   *
   * 
   */
  KOKKOS_INLINE_FUNCTION
  void slope_unsplit_hydro_2d(MHDState qNb[3][3],
			      MHDState (&dq)[2]) const
  {			
    real_t slope_type = params.settings.slope_type;

    // index of current cell in the neighborhood
    enum {CENTER=1};

    // aliases to input qState neighbors
    MHDState &q       = qNb[CENTER  ][CENTER  ];
    MHDState &qPlusX  = qNb[CENTER+1][CENTER  ];
    MHDState &qMinusX = qNb[CENTER-1][CENTER  ];
    MHDState &qPlusY  = qNb[CENTER  ][CENTER+1]; 
    MHDState &qMinusY = qNb[CENTER  ][CENTER-1];

    MHDState &dqX = dq[IX];
    MHDState &dqY = dq[IY];
 
    if (slope_type==1 or
	slope_type==2) {  // minmod or average

      slope_unsplit_hydro_2d_scalar(q.d,qPlusX.d,qMinusX.d,qPlusY.d,qMinusY.d, &(dqX.d), &(dqY.d));
      slope_unsplit_hydro_2d_scalar(q.p,qPlusX.p,qMinusX.p,qPlusY.p,qMinusY.p, &(dqX.p), &(dqY.p));
      slope_unsplit_hydro_2d_scalar(q.u,qPlusX.u,qMinusX.u,qPlusY.u,qMinusY.u, &(dqX.u), &(dqY.u));
      slope_unsplit_hydro_2d_scalar(q.v,qPlusX.v,qMinusX.v,qPlusY.v,qMinusY.v, &(dqX.v), &(dqY.v));
      slope_unsplit_hydro_2d_scalar(q.w,qPlusX.w,qMinusX.w,qPlusY.w,qMinusY.w, &(dqX.w), &(dqY.w));
      slope_unsplit_hydro_2d_scalar(q.bx,qPlusX.bx,qMinusX.bx,qPlusY.bx,qMinusY.bx, &(dqX.bx), &(dqY.bx));
      slope_unsplit_hydro_2d_scalar(q.by,qPlusX.by,qMinusX.by,qPlusY.by,qMinusY.by, &(dqX.by), &(dqY.by));
      slope_unsplit_hydro_2d_scalar(q.bz,qPlusX.bz,qMinusX.bz,qPlusY.bz,qMinusY.bz, &(dqX.bz), &(dqY.bz));
      
    }
    // else if (::gParams.slope_type == 3) {
    
    //   real_t slop, dlim;
    //   real_t dfll, dflm, dflr, dfml, dfmm, dfmr, dfrl, dfrm, dfrr;
    //   real_t vmin, vmax;
    //   real_t dfx, dfy, dff;

    //   for (int nVar=0; nVar<NVAR_MHD; ++nVar) {
    
    // 	dfll = qNb[CENTER-1][CENTER-1][nVar]-qNb[CENTER][CENTER][nVar];
    // 	dflm = qNb[CENTER-1][CENTER  ][nVar]-qNb[CENTER][CENTER][nVar];
    // 	dflr = qNb[CENTER-1][CENTER+1][nVar]-qNb[CENTER][CENTER][nVar];
    // 	dfml = qNb[CENTER  ][CENTER-1][nVar]-qNb[CENTER][CENTER][nVar];
    // 	dfmm = qNb[CENTER  ][CENTER  ][nVar]-qNb[CENTER][CENTER][nVar];
    // 	dfmr = qNb[CENTER  ][CENTER+1][nVar]-qNb[CENTER][CENTER][nVar];
    // 	dfrl = qNb[CENTER+1][CENTER-1][nVar]-qNb[CENTER][CENTER][nVar];
    // 	dfrm = qNb[CENTER+1][CENTER  ][nVar]-qNb[CENTER][CENTER][nVar];
    // 	dfrr = qNb[CENTER+1][CENTER+1][nVar]-qNb[CENTER][CENTER][nVar];
      
    // 	vmin = FMIN9_(dfll,dflm,dflr,dfml,dfmm,dfmr,dfrl,dfrm,dfrr);
    // 	vmax = FMAX9_(dfll,dflm,dflr,dfml,dfmm,dfmr,dfrl,dfrm,dfrr);
	
    // 	dfx  = HALF_F * (qNb[CENTER+1][CENTER  ][nVar] - qNb[CENTER-1][CENTER  ][nVar]);
    // 	dfy  = HALF_F * (qNb[CENTER  ][CENTER+1][nVar] - qNb[CENTER  ][CENTER-1][nVar]);
    // 	dff  = HALF_F * (FABS(dfx) + FABS(dfy));
	
    // 	if (dff>ZERO_F) {
    // 	  slop = FMIN(ONE_F, FMIN(FABS(vmin), FABS(vmax))/dff);
    // 	} else {
    // 	  slop = ONE_F;
    // 	}
      
    // 	dlim = slop;
      
    // 	dqX[nVar] = dlim*dfx;
    // 	dqY[nVar] = dlim*dfy;
      
    //   } // end for nVar
  
    // } // end slope_type
  
  } // slope_unsplit_hydro_2d

  /**
   * slope_unsplit_mhd_2d computes only magnetic field slopes in 2D; hydro
   * slopes are always computed in slope_unsplit_hydro_2d.
   * 
   * Compute magnetic field slopes (vector dbf) from bf (face-centered)
   * and its neighbors. 
   * 
   * Note that slope_type is a global variable, located in symbol memory when 
   * using the GPU version.
   *
   * Loosely adapted from RAMSES and DUMSES mhd/umuscl.f90: subroutine uslope
   * Interface is changed to become cellwise.
   *
   * \param[in]  bf  : face centered magnetic field in current
   * and neighboring cells. There are 6 values (3 values for bf_x along
   * y and 3 for bf_y along x).
   * 
   * \param[out] dbf : reference to an array returning magnetic field slopes 
   */
  KOKKOS_INLINE_FUNCTION
  void slope_unsplit_mhd_2d(real_t bfNeighbors[6],
			    real_t (&dbf)[2][3]) const
  {			
    /* layout for face centered magnetic field */
    real_t &bfx        = bfNeighbors[0];
    real_t &bfx_yplus  = bfNeighbors[1];
    real_t &bfx_yminus = bfNeighbors[2];
    real_t &bfy        = bfNeighbors[3];
    real_t &bfy_xplus  = bfNeighbors[4];
    real_t &bfy_xminus = bfNeighbors[5];
  
    real_t (&dbfX)[3] = dbf[IX];
    real_t (&dbfY)[3] = dbf[IY];

    // default values for magnetic field slopes
    for (int nVar=0; nVar<3; ++nVar) {
      dbfX[nVar] = ZERO_F;
      dbfY[nVar] = ZERO_F;
    }
  
    /*
     * face-centered magnetic field slopes
     */
    // 1D transverse TVD slopes for face-centered magnetic fields
  
    {
      // Bx along direction Y 
      real_t dlft, drgt, dcen, dsgn, slop, dlim;
      dlft = params.settings.slope_type * (bfx       - bfx_yminus);
      drgt = params.settings.slope_type * (bfx_yplus - bfx       );
      dcen = HALF_F * (bfx_yplus - bfx_yminus);
      dsgn = (dcen >= ZERO_F) ? ONE_F : -ONE_F;
      slop = FMIN( FABS(dlft), FABS(drgt) );
      dlim = slop;
      if ( (dlft*drgt) <= ZERO_F )
	dlim = ZERO_F;
      dbfY[IX] = dsgn * FMIN( dlim, FABS(dcen) );
      
      // By along direction X
      dlft = params.settings.slope_type * (bfy       - bfy_xminus);
      drgt = params.settings.slope_type * (bfy_xplus - bfy       );
      dcen = HALF_F * (bfy_xplus - bfy_xminus);
      dsgn = (dcen >= ZERO_F) ? ONE_F : -ONE_F;
      slop = FMIN( FABS(dlft), FABS(drgt) );
      dlim = slop;
      if( (dlft*drgt) <= ZERO_F )
	dlim=ZERO_F;
      dbfX[IY] = dsgn * FMIN( dlim, FABS(dcen) );
    }

  } // slope_unsplit_mhd_2d

  /**
   * Trace computations for unsplit Godunov scheme.
   *
   * \param[in] q          : Primitive variables state.
   * \param[in] dqX        : slope along X
   * \param[in] dqY        : slope along Y
   * \param[in] c          : local sound speed.
   * \param[in] dtdx       : dt over dx
   * \param[in] dtdy       : dt over dy
   * \param[in] faceId     : which face will be reconstructed
   * \param[out] qface     : q reconstructed state at cell interface
   */
  KOKKOS_INLINE_FUNCTION
  void trace_unsplit_2d_along_dir(const MHDState *q, 
				  const MHDState *dqX,
				  const MHDState *dqY,
				  real_t dtdx, 
				  real_t dtdy, 
				  int    faceId,
				  MHDState *qface) const
  {
  
    real_t gamma0 = params.settings.gamma0;
    real_t smallr = params.settings.smallr;

    // Cell centered values
    real_t r =  q->d;
    real_t p =  q->p;
    real_t u =  q->u;
    real_t v =  q->v;
  
    // TVD slopes in all directions
    real_t drx = dqX->d;
    real_t dpx = dqX->p;
    real_t dux = dqX->u;
    real_t dvx = dqX->v;
  
    real_t dry = dqY->d;
    real_t dpy = dqY->p;
    real_t duy = dqY->u;
    real_t dvy = dqY->v;
  
    // source terms (with transverse derivatives)
    real_t sr0 = -u*drx-v*dry - (dux+dvy)*r;
    real_t sp0 = -u*dpx-v*dpy - (dux+dvy)*gamma0*p;
    real_t su0 = -u*dux-v*duy - (dpx    )/r;
    real_t sv0 = -u*dvx-v*dvy - (dpy    )/r;
  
    if (faceId == FACE_XMIN) {
      // Right state at left interface
      qface->d = r - 0.5*drx + sr0*dtdx*0.5;
      qface->p = p - 0.5*dpx + sp0*dtdx*0.5;
      qface->u = u - 0.5*dux + su0*dtdx*0.5;
      qface->v = v - 0.5*dvx + sv0*dtdx*0.5;
      qface->d = fmax(smallr, qface->d);
    }

    if (faceId == FACE_XMAX) {
      // Left state at right interface
      qface->d = r + 0.5*drx + sr0*dtdx*0.5;
      qface->p = p + 0.5*dpx + sp0*dtdx*0.5;
      qface->u = u + 0.5*dux + su0*dtdx*0.5;
      qface->v = v + 0.5*dvx + sv0*dtdx*0.5;
      qface->d = fmax(smallr, qface->d);
    }
  
    if (faceId == FACE_YMIN) {
      // Top state at bottom interface
      qface->d = r - 0.5*dry + sr0*dtdy*0.5;
      qface->p = p - 0.5*dpy + sp0*dtdy*0.5;
      qface->u = u - 0.5*duy + su0*dtdy*0.5;
      qface->v = v - 0.5*dvy + sv0*dtdy*0.5;
      qface->d = fmax(smallr, qface->d);
    }

    if (faceId == FACE_YMAX) {
      // Bottom state at top interface
      qface->d = r + 0.5*dry + sr0*dtdy*0.5;
      qface->p = p + 0.5*dpy + sp0*dtdy*0.5;
      qface->u = u + 0.5*duy + su0*dtdy*0.5;
      qface->v = v + 0.5*dvy + sv0*dtdy*0.5;
      qface->d = fmax(smallr, qface->d);
    }

  } // trace_unsplit_2d_along_dir


  /**
   * This another implementation of trace computations for 2D data; it
   * is used when unsplitVersion = 1
   *
   * Note that :
   * - hydro slopes computations are done outside this routine
   *
   * \param[in]  q  primitive variable state vector
   * \param[in]  dq primitive variable slopes
   * \param[in]  dtdx dt divided by dx
   * \param[in]  dtdy dt divided by dy
   * \param[out] qm
   * \param[out] qp
   *
   */
  KOKKOS_INLINE_FUNCTION
  void trace_unsplit_hydro_2d(const MHDState *q,
			      const MHDState *dqX,
			      const MHDState *dqY,
			      real_t dtdx,
			      real_t dtdy,
			      MHDState *qm_x,
			      MHDState *qm_y,
			      MHDState *qp_x,
			      MHDState *qp_y) const
  {
  
    real_t gamma0 = params.settings.gamma0;
    real_t smallr = params.settings.smallr;
    real_t smallp = params.settings.smallp;

    // Cell centered values
    real_t r = q->d;
    real_t p = q->p;
    real_t u = q->u;
    real_t v = q->v;

    // Cell centered TVD slopes in X direction
    real_t drx = dqX->d;  drx *= 0.5;
    real_t dpx = dqX->p;  dpx *= 0.5;
    real_t dux = dqX->u;  dux *= 0.5;
    real_t dvx = dqX->v;  dvx *= 0.5;
  
    // Cell centered TVD slopes in Y direction
    real_t dry = dqY->d;  dry *= 0.5;
    real_t dpy = dqY->p;  dpy *= 0.5;
    real_t duy = dqY->u;  duy *= 0.5;
    real_t dvy = dqY->v;  dvy *= 0.5;

    // Source terms (including transverse derivatives)
    real_t sr0, su0, sv0, sp0;

    /*only true for cartesian grid */
    {
      sr0 = (-u*drx-dux*r)       *dtdx + (-v*dry-dvy*r)       *dtdy;
      su0 = (-u*dux-dpx/r)       *dtdx + (-v*duy      )       *dtdy;
      sv0 = (-u*dvx      )       *dtdx + (-v*dvy-dpy/r)       *dtdy;
      sp0 = (-u*dpx-dux*gamma0*p)*dtdx + (-v*dpy-dvy*gamma0*p)*dtdy;    
    } // end cartesian

    // Update in time the  primitive variables
    r = r + sr0;
    u = u + su0;
    v = v + sv0;
    p = p + sp0;

    // Face averaged right state at left interface
    qp_x->d = r - drx;
    qp_x->u = u - dux;
    qp_x->v = v - dvx;
    qp_x->p = p - dpx;
    qp_x->d = fmax(smallr,  qp_x->d);
    qp_x->p = fmax(smallp * qp_x->d, qp_x->p);
  
    // Face averaged left state at right interface
    qm_x->d = r + drx;
    qm_x->u = u + dux;
    qm_x->v = v + dvx;
    qm_x->p = p + dpx;
    qm_x->d = fmax(smallr,  qm_x->d);
    qm_x->p = fmax(smallp * qm_x->d, qm_x->p);

    // Face averaged top state at bottom interface
    qp_y->d = r - dry;
    qp_y->u = u - duy;
    qp_y->v = v - dvy;
    qp_y->p = p - dpy;
    qp_y->d = fmax(smallr,  qp_y->d);
    qp_y->p = fmax(smallp * qp_y->d, qp_y->p);
  
    // Face averaged bottom state at top interface
    qm_y->d = r + dry;
    qm_y->u = u + duy;
    qm_y->v = v + dvy;
    qm_y->p = p + dpy;
    qm_y->d = fmax(smallr,  qm_y->d);
    qm_y->p = fmax(smallp * qm_y->d, qm_y->p);
  
  } // trace_unsplit_hydro_2d

  /**
   * 2D Trace computations for unsplit Godunov scheme.
   *
   * \note Note that this routine uses global variables iorder, scheme and
   * slope_type.
   *
   * \note Note that is routine is loosely adapted from trace2d found in 
   * Dumses and in Ramses sources (sub-dir mhd, file umuscl.f90) to be now a one cell 
   * computation. 
   *
   * \param[in]  qNb        state in neighbor cells (3-by-3 neighborhood indexed as qNb[i][j], for i,j=0,1,2); current center cell is at index (i=j=1).
   * \param[in]  bfNb       face centered magnetic field in neighbor cells (4-by-4 neighborhood indexed as bfNb[i][j] for i,j=0,1,2,3); current cell is located at index (i=j=1)
   * \param[in]  c          local sound speed.
   * \param[in]  dtdx       dt over dx
   * \param[in]  dtdy       dt over dy
   * \param[in]  xPos       x location of current cell (needed for shear computation)
   * \param[out] qm         qm state (one per dimension)
   * \param[out] qp         qp state (one per dimension)
   * \param[out] qEdge      q state on cell edges (qRT, qRB, qLT, qLB)
   */
  KOKKOS_INLINE_FUNCTION
  void trace_unsplit_mhd_2d(MHDState qNb[3][3],
			    BField bfNb[4][4],
			    real_t c, 
			    real_t dtdx,
			    real_t dtdy,
			    real_t xPos,
			    MHDState (&qm)[2], 
			    MHDState (&qp)[2],
			    MHDState (&qEdge)[4]) const
  {
    (void) c;

    // neighborhood sizes
    enum {Q_SIZE=3, BF_SIZE = 4};

    // index of current cell in the neighborhood
    enum {CENTER=1};

    // alias for q on cell edge (as defined in DUMSES trace2d routine)
    MHDState &qRT = qEdge[0];
    MHDState &qRB = qEdge[1];
    MHDState &qLT = qEdge[2];
    MHDState &qLB = qEdge[3];

    real_t smallR = params.settings.smallr;
    real_t smallp = params.settings.smallp;
    //real_t &smallP = params.settings.smallpp;
    real_t gamma  = params.settings.gamma0;
    //real_t &Omega0 = params.settings.Omega0;

    MHDState &q = qNb[CENTER][CENTER]; // current cell (neighborhood center)

    // compute u,v,A,B,Ez (electric field)
    real_t Ez[2][2];
    for (int di=0; di<2; di++)
      for (int dj=0; dj<2; dj++) {
      
	int centerX = CENTER+di;
	int centerY = CENTER+dj;
	real_t u  = 0.25f *  (qNb[centerX-1][centerY-1].u + 
			      qNb[centerX-1][centerY  ].u + 
			      qNb[centerX  ][centerY-1].u + 
			      qNb[centerX  ][centerY  ].u); 
      
	real_t v  = 0.25f *  (qNb[centerX-1][centerY-1].v +
			      qNb[centerX-1][centerY  ].v +
			      qNb[centerX  ][centerY-1].v + 
			      qNb[centerX  ][centerY  ].v);
      
	real_t A  = 0.5f  * (bfNb[centerX  ][centerY-1].bx + 
			     bfNb[centerX  ][centerY  ].bx);

	real_t B  = 0.5f  * (bfNb[centerX-1][centerY  ].by + 
			     bfNb[centerX  ][centerY  ].by);
      
	Ez[di][dj] = u*B-v*A;
      }

    // Electric field
    real_t &ELL = Ez[0][0];
    real_t &ELR = Ez[0][1];
    real_t &ERL = Ez[1][0];
    real_t &ERR = Ez[1][1];

    // Cell centered values
    real_t r = q.d;
    real_t p = q.p;
    real_t u = q.u;
    real_t v = q.v;
    real_t w = q.w;            
    real_t A = q.bx;
    real_t B = q.by;
    real_t C = q.bz;            
    
    // Face centered variables
    real_t AL =  bfNb[CENTER  ][CENTER  ].bx;
    real_t AR =  bfNb[CENTER+1][CENTER  ].bx;
    real_t BL =  bfNb[CENTER  ][CENTER  ].by;
    real_t BR =  bfNb[CENTER  ][CENTER+1].by;

    // TODO LATER : compute xL, xR and xC using ::gParam
    // this is only needed when doing cylindrical or spherical coordinates

    /*
     * compute dq slopes
     */
    MHDState dq[2];

    slope_unsplit_hydro_2d(qNb, dq);

    // slight modification compared to DUMSES (we re-used dq itself,
    // instead of re-declaring new variables, better for the GPU
    // register count

    // Cell centered TVD slopes in X direction
    real_t drx = dq[IX].d;  drx *= 0.5;
    real_t dpx = dq[IX].p;  dpx *= 0.5;
    real_t dux = dq[IX].u;  dux *= 0.5;
    real_t dvx = dq[IX].v;  dvx *= 0.5;
    real_t dwx = dq[IX].w;  dwx *= 0.5;
    real_t dCx = dq[IX].bz;  dCx *= 0.5;
    real_t dBx = dq[IX].by;  dBx *= 0.5;
  
    // Cell centered TVD slopes in Y direction
    real_t dry = dq[IY].d;  dry *= 0.5;
    real_t dpy = dq[IY].p;  dpy *= 0.5;
    real_t duy = dq[IY].u;  duy *= 0.5;
    real_t dvy = dq[IY].v;  dvy *= 0.5;
    real_t dwy = dq[IY].w;  dwy *= 0.5;
    real_t dCy = dq[IY].bz;  dCy *= 0.5;
    real_t dAy = dq[IY].bx;  dAy *= 0.5;
  
    /*
     * compute dbf slopes needed for Face centered TVD slopes in transverse direction
     */
    real_t bfNeighbors[6];
    real_t dbf[2][3];
    real_t (&dbfX)[3] = dbf[IX];
    real_t (&dbfY)[3] = dbf[IY];
  
    bfNeighbors[0] =  bfNb[CENTER  ][CENTER  ].bx;
    bfNeighbors[1] =  bfNb[CENTER  ][CENTER+1].bx;
    bfNeighbors[2] =  bfNb[CENTER  ][CENTER-1].bx;
    bfNeighbors[3] =  bfNb[CENTER  ][CENTER  ].by;
    bfNeighbors[4] =  bfNb[CENTER+1][CENTER  ].by;
    bfNeighbors[5] =  bfNb[CENTER-1][CENTER  ].by;
  
    slope_unsplit_mhd_2d(bfNeighbors, dbf);
  
    // Face centered TVD slopes in transverse direction
    real_t dALy = 0.5 * dbfY[IX];
    real_t dBLx = 0.5 * dbfX[IY];

    // change neighbors to i+1, j and recompute dbf
    bfNeighbors[0] =  bfNb[CENTER+1][CENTER  ].bx;
    bfNeighbors[1] =  bfNb[CENTER+1][CENTER+1].bx;
    bfNeighbors[2] =  bfNb[CENTER+1][CENTER-1].bx;
    bfNeighbors[3] =  bfNb[CENTER+1][CENTER  ].by;
    bfNeighbors[4] =  bfNb[CENTER+2][CENTER  ].by;
    bfNeighbors[5] =  bfNb[CENTER  ][CENTER  ].by;

    slope_unsplit_mhd_2d(bfNeighbors, dbf);  

    real_t dARy = 0.5 * dbfY[IX];

    // change neighbors to i, j+1 and recompute dbf
    bfNeighbors[0] =  bfNb[CENTER  ][CENTER+1].bx;
    bfNeighbors[1] =  bfNb[CENTER  ][CENTER+2].bx;
    bfNeighbors[2] =  bfNb[CENTER  ][CENTER  ].bx;
    bfNeighbors[3] =  bfNb[CENTER  ][CENTER+1].by;
    bfNeighbors[4] =  bfNb[CENTER+1][CENTER+1].by;
    bfNeighbors[5] =  bfNb[CENTER-1][CENTER+1].by;

    slope_unsplit_mhd_2d(bfNeighbors, dbf);

    real_t dBRx = 0.5 * dbfX[IY];
  
    // Cell centered slopes in normal direction
    real_t dAx = 0.5 * (AR - AL);
    real_t dBy = 0.5 * (BR - BL);
  
    // Source terms (including transverse derivatives)
    real_t sr0, su0, sv0, sw0, sp0, sA0, sB0, sC0;
    real_t sAL0, sAR0, sBL0, sBR0;

    if (true /*cartesian*/) {

      sr0 = (-u*drx-dux*r)                *dtdx + (-v*dry-dvy*r)                *dtdy;
      su0 = (-u*dux-dpx/r-B*dBx/r-C*dCx/r)*dtdx + (-v*duy+B*dAy/r)              *dtdy;
      sv0 = (-u*dvx+A*dBx/r)              *dtdx + (-v*dvy-dpy/r-A*dAy/r-C*dCy/r)*dtdy;
      sw0 = (-u*dwx+A*dCx/r)              *dtdx + (-v*dwy+B*dCy/r)              *dtdy;
      sp0 = (-u*dpx-dux*gamma*p)          *dtdx + (-v*dpy-dvy*gamma*p)          *dtdy;
      sA0 =                                       ( u*dBy+B*duy-v*dAy-A*dvy)    *dtdy;
      sB0 = (-u*dBx-B*dux+v*dAx+A*dvx)    *dtdx ;
      sC0 = ( w*dAx+A*dwx-u*dCx-C*dux)    *dtdx + (-v*dCy-C*dvy+w*dBy+B*dwy)    *dtdy;
      // if (Omega0 > ZERO_F) {
      // 	real_t shear = -1.5 * Omega0 * xPos;
      // 	sC0 += (shear * dAx - 1.5 * Omega0 * A) * dtdx;
      // 	sC0 +=  shear * dBy                     * dtdy;
      // }

      // Face centered B-field
      sAL0 = +(ELR-ELL)*0.5*dtdy;
      sAR0 = +(ERR-ERL)*0.5*dtdy;
      sBL0 = -(ERL-ELL)*0.5*dtdx;
      sBR0 = -(ERR-ELR)*0.5*dtdx;

    } // end cartesian
  
    // Update in time the  primitive variables
    r = r + sr0;
    u = u + su0;
    v = v + sv0;
    w = w + sw0;
    p = p + sp0;
    A = A + sA0;
    B = B + sB0;
    C = C + sC0;
  
    AL = AL + sAL0;
    AR = AR + sAR0;
    BL = BL + sBL0;
    BR = BR + sBR0;
  
    // Right state at left interface
    qp[0].d = r - drx;
    qp[0].u = u - dux;
    qp[0].v = v - dvx;
    qp[0].w = w - dwx;
    qp[0].p = p - dpx;
    qp[0].bx = AL;
    qp[0].by = B - dBx;
    qp[0].bz = C - dCx;
    qp[0].d = FMAX(smallR, qp[0].d);
    qp[0].p = FMAX(smallp*qp[0].d, qp[0].p);
  
    // Left state at right interface
    qm[0].d = r + drx;
    qm[0].u = u + dux;
    qm[0].v = v + dvx;
    qm[0].w = w + dwx;
    qm[0].p = p + dpx;
    qm[0].bx = AR;
    qm[0].by = B + dBx;
    qm[0].bz = C + dCx;
    qm[0].d = FMAX(smallR, qm[0].d);
    qm[0].p = FMAX(smallp*qm[0].d, qm[0].p);
  
    // Top state at bottom interface
    qp[1].d = r - dry;
    qp[1].u = u - duy;
    qp[1].v = v - dvy;
    qp[1].w = w - dwy;
    qp[1].p = p - dpy;
    qp[1].bx = A - dAy;
    qp[1].by = BL;
    qp[1].bz = C - dCy;
    qp[1].d = FMAX(smallR, qp[1].d);
    qp[1].p = FMAX(smallp*qp[1].d, qp[1].p);
  
    // Bottom state at top interface
    qm[1].d = r + dry;
    qm[1].u = u + duy;
    qm[1].v = v + dvy;
    qm[1].w = w + dwy;
    qm[1].p = p + dpy;
    qm[1].bx = A + dAy;
    qm[1].by = BR;
    qm[1].bz = C + dCy;
    qm[1].d = FMAX(smallR, qm[1].d);
    qm[1].p = FMAX(smallp*qm[1].d, qm[1].p);
  
  
    // Right-top state (RT->LL)
    qRT.d = r + (+drx+dry);
    qRT.u = u + (+dux+duy);
    qRT.v = v + (+dvx+dvy);
    qRT.w = w + (+dwx+dwy);
    qRT.p = p + (+dpx+dpy);
    qRT.bx = AR+ (   +dARy);
    qRT.by = BR+ (+dBRx   );
    qRT.bz = C + (+dCx+dCy);
    qRT.d = FMAX(smallR, qRT.d);
    qRT.p = FMAX(smallp*qRT.d, qRT.p);
    
    // Right-Bottom state (RB->LR)
    qRB.d = r + (+drx-dry);
    qRB.u = u + (+dux-duy);
    qRB.v = v + (+dvx-dvy);
    qRB.w = w + (+dwx-dwy);
    qRB.p = p + (+dpx-dpy);
    qRB.bx = AR+ (   -dARy);
    qRB.by = BL+ (+dBLx   );
    qRB.bz = C + (+dCx-dCy);
    qRB.d = FMAX(smallR, qRB.d);
    qRB.p = FMAX(smallp*qRB.d, qRB.p);
    
    // Left-Bottom state (LB->RR)
    qLB.d = r + (-drx-dry);
    qLB.u = u + (-dux-duy);
    qLB.v = v + (-dvx-dvy);
    qLB.w = w + (-dwx-dwy);
    qLB.p = p + (-dpx-dpy);
    qLB.bx = AL+ (   -dALy);
    qLB.by = BL+ (-dBLx   );
    qLB.bz = C + (-dCx-dCy);
    qLB.d = FMAX(smallR, qLB.d);
    qLB.p = FMAX(smallp*qLB.d, qLB.p);
    
    // Left-Top state (LT->RL)
    qLT.d = r + (-drx+dry);
    qLT.u = u + (-dux+duy);
    qLT.v = v + (-dvx+dvy);
    qLT.w = w + (-dwx+dwy);
    qLT.p = p + (-dpx+dpy);
    qLT.bx = AL+ (   +dALy);
    qLT.by = BR+ (-dBRx   );
    qLT.bz = C + (-dCx+dCy);
    qLT.d = FMAX(smallR, qLT.d);
    qLT.p = FMAX(smallp*qLT.d, qLT.p);

  } // trace_unsplit_mhd_2d

  /**
   * Compute primitive variables slope (vector dq) from q and its neighbors.
   * This routine is only used in the 2D UNSPLIT integration and slope_type = 0,1 and 2.
   * 
   * Only slope_type 1 and 2 are supported.
   *
   * \param[in]  q       : current primitive variable state
   * \param[in]  qPlusX  : state in the next neighbor cell along XDIR
   * \param[in]  qMinusX : state in the previous neighbor cell along XDIR
   * \param[in]  qPlusY  : state in the next neighbor cell along YDIR
   * \param[in]  qMinusY : state in the previous neighbor cell along YDIR
   * \param[out] dqX     : reference to an array returning the X slopes
   * \param[out] dqY     : reference to an array returning the Y slopes
   *
   */
  KOKKOS_INLINE_FUNCTION
  void slope_unsplit_hydro_2d(const MHDState *q, 
			      const MHDState *qPlusX, 
			      const MHDState *qMinusX,
			      const MHDState *qPlusY,
			      const MHDState *qMinusY,
			      MHDState *dqX,
			      MHDState *dqY) const
  {
  
    real_t slope_type = params.settings.slope_type;

    if (slope_type==0) {

      dqX->d = ZERO_F;
      dqX->p = ZERO_F;
      dqX->u = ZERO_F;
      dqX->v = ZERO_F;

      dqY->d = ZERO_F;
      dqY->p = ZERO_F;
      dqY->u = ZERO_F;
      dqY->v = ZERO_F;

      return;
    }

    if (slope_type==1 || slope_type==2) {  // minmod or average

      slope_unsplit_hydro_2d_scalar( q->d, qPlusX->d, qMinusX->d, qPlusY->d, qMinusY->d, &(dqX->d), &(dqY->d));
      slope_unsplit_hydro_2d_scalar( q->p, qPlusX->p, qMinusX->p, qPlusY->p, qMinusY->p, &(dqX->p), &(dqY->p));
      slope_unsplit_hydro_2d_scalar( q->u, qPlusX->u, qMinusX->u, qPlusY->u, qMinusY->u, &(dqX->u), &(dqY->u));
      slope_unsplit_hydro_2d_scalar( q->v, qPlusX->v, qMinusX->v, qPlusY->v, qMinusY->v, &(dqX->v), &(dqY->v));

    } // end slope_type == 1 or 2
  
  } // slope_unsplit_hydro_2d

  /**
   * Compute cell fluxes from the Godunov state
   * \param[in]  qgdnv input Godunov state
   * \param[out] flux  output flux vector
   */
  KOKKOS_INLINE_FUNCTION
  void cmpflx(const MHDState *qgdnv, 
	      MHDState *flux) const
  {
    real_t gamma0 = params.settings.gamma0;

    // Compute fluxes
    // Mass density
    flux->d = qgdnv->d * qgdnv->u;
  
    // Normal momentum
    flux->u = flux->d * qgdnv->u + qgdnv->p;
  
    // Transverse momentum
    flux->v = flux->d * qgdnv->v;

    // Total energy
    real_t entho = ONE_F / (gamma0 - ONE_F);
    real_t ekin;
    ekin = 0.5 * qgdnv->d * (qgdnv->u*qgdnv->u + qgdnv->v*qgdnv->v);
  
    real_t etot = qgdnv->p * entho + ekin;
    flux->p = qgdnv->u * (etot + qgdnv->p);

  } // cmpflx
  
  /**
   * Computes fastest signal speed for each direction.
   *
   * \param[in]  qState       primitive variables state vector
   * \param[out] fastInfoSpeed array containing fastest information speed along
   * x, y, and z direction.
   *
   * Directionnal information speed being defined as :
   * directionnal fast magneto speed + FABS(velocity component)
   *
   * \warning This routine uses gamma ! You need to set gamma to something very near to 1
   *
   * \tparam nDim if nDim==2, only computes information speed along x
   * and y.
   */
  template<DimensionType nDim>
  KOKKOS_INLINE_FUNCTION
  void find_speed_info(const MHDState& qState, 
		       real_t (&fastInfoSpeed)[3]) const
  {
    
    real_t gamma0  = params.settings.gamma0;
    double d,p,a,b,c,b2,c2,d2,cf;
    double u = qState.u;
    double v = qState.v;
    double w = qState.w;
    
    d=qState.d;  p=qState.p; 
    a=qState.bx; b=qState.by; c=qState.bz;
    
    /*
     * compute fastest info speed along X
     */
    
    // square norm of magnetic field
    b2 = a*a + b*b + c*c;
    
    // square speed of sound
    c2 = gamma0 * p / d;
    
    d2 = 0.5 * (b2/d + c2);
    
    cf = SQRT( d2 + SQRT(d2*d2 - c2*a*a/d) );
    
    fastInfoSpeed[IX] = cf+fabs(u);
    
    // compute fastest info speed along Y
    cf = SQRT( d2 + sqrt(d2*d2 - c2*b*b/d) );
    
    fastInfoSpeed[IY] = cf+fabs(v);
    
    
    // compute fastest info speed along Z
    if (nDim == THREE_D) {
      cf = sqrt( d2 + SQRT(d2*d2 - c2*c*c/d) );
      
      fastInfoSpeed[IZ] = cf+fabs(w);
    } // end THREE_D
    
  } // find_speed_info
  
  /**
   * Compute the fast magnetosonic velocity.
   * 
   * IU is index to Vnormal
   * IA is index to Bnormal
   * 
   * IV, IW are indexes to Vtransverse1, Vtransverse2,
   * IB, IC are indexes to Btransverse1, Btransverse2
   *
   */
  template <ComponentIndex3D dir>
  KOKKOS_INLINE_FUNCTION
  real_t find_speed_fast(const MHDState& qvar) const
  {
    
    real_t gamma0  = params.settings.gamma0;
    real_t d,p,a,b,c,b2,c2,d2,cf;
    
    d=qvar.d;  p=qvar.p; 
    a=qvar.bx; b=qvar.by; c=qvar.bz;
    
    b2 = a*a + b*b + c*c;
    c2 = gamma0 * p / d;
    d2 = 0.5 * (b2/d + c2);
    if (dir==IX)
      cf = sqrt( d2 + sqrt(d2*d2 - c2*a*a/d) );
    
    if (dir==IY)
      cf = sqrt( d2 + sqrt(d2*d2 - c2*b*b/d) );
    
    if (dir==IZ)
      cf = sqrt( d2 + sqrt(d2*d2 - c2*c*c/d) );
    
    return cf;
    
  } // find_speed_fast
      
  /**
   * Compute the 1d mhd fluxes from the conservative.
   *
   * Only used in Riemann solver HLL (probably cartesian only
   * compatible, since gas pressure is included).
   *
   * variables. The structure of qvar is : 
   * rho, pressure,
   * vnormal, vtransverse1, vtransverse2, 
   * bnormal, btransverse1, btransverse2.
   *
   * @param[in]  qvar state vector (primitive variables)
   * @param[out] cvar state vector (conservative variables)
   * @param[out] ff flux vector
   *
   */
  KOKKOS_INLINE_FUNCTION
  void find_mhd_flux(MHDState qvar, 
		     MHDState &cvar,
		     MHDState &ff) const
  {
    
    // ISOTHERMAL
    real_t cIso = params.settings.cIso;
    real_t p;
    if (cIso>0) {
      // recompute pressure
      p = qvar.d*cIso*cIso;
    } else {
      p = qvar.p;
    }
    // end ISOTHERMAL
    
    // local variables
    const real_t entho = ONE_F / (params.settings.gamma0 - ONE_F);
    
    real_t d, u, v, w, a, b, c;
    d=qvar.d; 
    u=qvar.u; v=qvar.v; w=qvar.w;
    a=qvar.bx; b=qvar.by; c=qvar.bz;
    
    real_t ecin = 0.5*(u*u+v*v+w*w)*d;
    real_t emag = 0.5*(a*a+b*b+c*c);
    real_t etot = p*entho+ecin+emag;
    real_t ptot = p + emag;
    
    // compute conservative variables
    cvar.d  = d;
    cvar.p  = etot;
    cvar.u  = d*u;
    cvar.v  = d*v;
    cvar.w  = d*w;
    cvar.bx = a;
    cvar.by = b;
    cvar.bz = c;
    
    // compute fluxes
    ff.d  = d*u;
    ff.p  = (etot+ptot)*u-a*(a*u+b*v+c*w);
    ff.u  = d*u*u-a*a+ptot; /* *** WARNING pressure included *** */
    ff.v  = d*u*v-a*b;
    ff.w  = d*u*w-a*c;
    ff.bx = 0.0;
    ff.by = b*u-a*v;
    ff.bz = c*u-a*w;
    
  } // find_mhd_flux

  /**
   * MHD HLL Riemann solver
   *
   * qleft, qright and flux have now NVAR_MHD=8 components.
   *
   * The following code is adapted from Dumses.
   *
   * @param[in] qleft  : input left state
   * @param[in] qright : input right state
   * @param[out] flux  : output flux
   *
   */
  KOKKOS_INLINE_FUNCTION
  void riemann_hll(MHDState &qleft,
		   MHDState &qright,
		   MHDState &flux) const
  {
    
    // enforce continuity of normal component
    real_t bx_mean = 0.5 * ( qleft.bx + qright.bx );
    qleft.bx  = bx_mean;
    qright.bx = bx_mean;
    
    MHDState uleft,  fleft;
    MHDState uright, fright;
    
    find_mhd_flux(qleft ,uleft ,fleft );
    find_mhd_flux(qright,uright,fright);
    
    // find the largest eigenvalue in the normal direction to the interface
    real_t cfleft  = find_speed_fast<IX>(qleft);
    real_t cfright = find_speed_fast<IX>(qright);
    
    real_t vleft =qleft.u;
    real_t vright=qright.u;
    real_t sl=fmin ( fmin (vleft,vright) - fmax (cfleft,cfright) , 0.0);
    real_t sr=fmax ( fmax (vleft,vright) + fmax (cfleft,cfright) , 0.0);
    
    // the hll flux
    flux.d = (sr*fleft.d-sl*fright.d+
	      sr*sl*(uright.d-uleft.d))/(sr-sl);
    flux.p = (sr*fleft.p-sl*fright.p+
	      sr*sl*(uright.p-uleft.p))/(sr-sl);
    flux.u = (sr*fleft.u-sl*fright.u+
	      sr*sl*(uright.u-uleft.u))/(sr-sl);
    flux.v = (sr*fleft.v-sl*fright.v+
	      sr*sl*(uright.v-uleft.v))/(sr-sl);
    flux.w = (sr*fleft.w-sl*fright.w+
	      sr*sl*(uright.w-uleft.w))/(sr-sl);
    flux.bx = (sr*fleft.bx-sl*fright.bx+
	       sr*sl*(uright.bx-uleft.bx))/(sr-sl);
    flux.by = (sr*fleft.by-sl*fright.by+
	       sr*sl*(uright.by-uleft.by))/(sr-sl);
    flux.bz = (sr*fleft.bz-sl*fright.bz+
	       sr*sl*(uright.bz-uleft.bz))/(sr-sl);

    
  } // riemann_hll

  /** 
   * Riemann solver, equivalent to riemann_hlld in RAMSES/DUMSES (see file
   * godunov_utils.f90 in RAMSES/DUMSES).
   *
   * Reference :
   * <A HREF="http://www.sciencedirect.com/science/article/B6WHY-4FY3P80-7/2/426234268c96dcca8a828d098b75fe4e">
   * Miyoshi & Kusano, 2005, JCP, 208, 315 </A>
   *
   * \warning This version of HLLD integrates the pressure term in
   * flux.u (as in RAMSES). This will need to be modified in the
   * future (as it is done in DUMSES) to handle cylindrical / spherical
   * coordinate systems. For example, one could add a new ouput named qStar
   * to store star state, and that could be used to compute geometrical terms
   * outside this routine.
   *
   * @param[in] qleft : input left state
   * @param[in] qright : input right state
   * @param[out] flux  : output flux
   */
  KOKKOS_INLINE_FUNCTION
  void riemann_hlld(MHDState &qleft,
		    MHDState &qright,
		    MHDState &flux) const
  {
    
    // Constants
    const real_t gamma0 = params.settings.gamma0;
    const real_t entho = 1.0 / (gamma0 - 1.0);
    
    // Enforce continuity of normal component of magnetic field
    real_t a    = 0.5 * ( qleft.bx + qright.bx );
    real_t sgnm = (a >= 0) ? ONE_F : -ONE_F;
    qleft .bx  = a; 
    qright.bx  = a;
    
    // ISOTHERMAL
    real_t cIso = params.settings.cIso;
    if (cIso > 0) {
      // recompute pressure
      qleft .p = qleft .d*cIso*cIso;
      qright.p = qright.d*cIso*cIso;
    } // end ISOTHERMAL
    
    // left variables
    real_t rl, pl, ul, vl, wl, bl, cl;
    rl = qleft.d; //rl = fmax(qleft.d, static_cast<real_t>(gParams.smallr)    );  
    pl = qleft.p; //pl = fmax(qleft.p, static_cast<real_t>(rl*gParams.smallp) ); 
    ul = qleft.u;  vl = qleft.v;  wl = qleft.w; 
    bl = qleft.by;  cl = qleft.bz;
    real_t ecinl = 0.5 * (ul*ul + vl*vl + wl*wl) * rl;
    real_t emagl = 0.5 * ( a*a  + bl*bl + cl*cl);
    real_t etotl = pl*entho + ecinl + emagl;
    real_t ptotl = pl + emagl;
    real_t vdotbl= ul*a + vl*bl + wl*cl;
    
    // right variables
    real_t rr, pr, ur, vr, wr, br, cr;
    rr = qright.d; //rr = fmax(qright.d, static_cast<real_t>( gParams.smallr) );
    pr = qright.p; //pr = fmax(qright.p, static_cast<real_t>( rr*gParams.smallp) ); 
    ur = qright.u;  vr=qright.v;  wr = qright.w; 
    br = qright.by;  cr=qright.bz;
    real_t ecinr = 0.5 * (ur*ur + vr*vr + wr*wr) * rr;
    real_t emagr = 0.5 * ( a*a  + br*br + cr*cr);
    real_t etotr = pr*entho + ecinr + emagr;
    real_t ptotr = pr + emagr;
    real_t vdotbr= ur*a + vr*br + wr*cr;
    
    // find the largest eigenvalues in the normal direction to the interface
    real_t cfastl = find_speed_fast<IX>(qleft);
    real_t cfastr = find_speed_fast<IX>(qright);
    
    // compute hll wave speed
    real_t sl = fmin(ul,ur) - fmax(cfastl,cfastr);
    real_t sr = fmax(ul,ur) + fmax(cfastl,cfastr);
    
    // compute lagrangian sound speed
    real_t rcl = rl * (ul-sl);
    real_t rcr = rr * (sr-ur);
    
    // compute acoustic star state
    real_t ustar   = (rcr*ur   +rcl*ul   +  (ptotl-ptotr))/(rcr+rcl);
    real_t ptotstar= (rcr*ptotl+rcl*ptotr+rcl*rcr*(ul-ur))/(rcr+rcl);
    
    // left star region variables
    real_t estar;
    real_t rstarl, el;
    rstarl = rl*(sl-ul)/(sl-ustar);
    estar  = rl*(sl-ul)*(sl-ustar)-a*a;
    el     = rl*(sl-ul)*(sl-ul   )-a*a;
    real_t vstarl, wstarl;
    real_t bstarl, cstarl;
    // not very good (should use a small energy cut-off !!!)
    if(a*a>0 and fabs(estar/(a*a)-ONE_F)<=1e-8) {
      vstarl=vl;
      bstarl=bl;
      wstarl=wl;
      cstarl=cl;
    } else {
      vstarl=vl-a*bl*(ustar-ul)/estar;
      bstarl=bl*el/estar;
      wstarl=wl-a*cl*(ustar-ul)/estar;
      cstarl=cl*el/estar;
    }
    real_t vdotbstarl = ustar*a+vstarl*bstarl+wstarl*cstarl;
    real_t etotstarl  = ((sl-ul)*etotl-ptotl*ul+ptotstar*ustar+a*(vdotbl-vdotbstarl))/(sl-ustar);
    real_t sqrrstarl  = sqrt(rstarl);
    real_t calfvenl   = fabs(a)/sqrrstarl; /* sqrrstarl should never be zero, but it might happen if border conditions are not OK !!!!!! */
    real_t sal        = ustar-calfvenl;
    
    // right star region variables
    real_t rstarr, er;
    rstarr = rr*(sr-ur)/(sr-ustar);
    estar  = rr*(sr-ur)*(sr-ustar)-a*a;
    er     = rr*(sr-ur)*(sr-ur   )-a*a;
    real_t vstarr, wstarr;
    real_t bstarr, cstarr;
    // not very good (should use a small energy cut-off !!!)
    if(a*a>0 and FABS(estar/(a*a)-ONE_F)<=1e-8) {
      vstarr=vr;
      bstarr=br;
      wstarr=wr;
      cstarr=cr;
    } else {
      vstarr=vr-a*br*(ustar-ur)/estar;
      bstarr=br*er/estar;
      wstarr=wr-a*cr*(ustar-ur)/estar;
      cstarr=cr*er/estar;
    }
    real_t vdotbstarr = ustar*a+vstarr*bstarr+wstarr*cstarr;
    real_t etotstarr  = ((sr-ur)*etotr-ptotr*ur+ptotstar*ustar+a*(vdotbr-vdotbstarr))/(sr-ustar);
    real_t sqrrstarr  = sqrt(rstarr);
    real_t calfvenr   = fabs(a)/sqrrstarr; /* sqrrstarr should never be zero, but it might happen if border conditions are not OK !!!!!! */
    real_t sar        = ustar+calfvenr;
    
    // double star region variables
    real_t vstarstar     = (sqrrstarl*vstarl+sqrrstarr*vstarr+
			    sgnm*(bstarr-bstarl)) / (sqrrstarl+sqrrstarr);
    real_t wstarstar     = (sqrrstarl*wstarl+sqrrstarr*wstarr+
			    sgnm*(cstarr-cstarl)) / (sqrrstarl+sqrrstarr);
    real_t bstarstar     = (sqrrstarl*bstarr+sqrrstarr*bstarl+
			    sgnm*sqrrstarl*sqrrstarr*(vstarr-vstarl)) / 
      (sqrrstarl+sqrrstarr);
    real_t cstarstar     = (sqrrstarl*cstarr+sqrrstarr*cstarl+
			    sgnm*sqrrstarl*sqrrstarr*(wstarr-wstarl)) /
      (sqrrstarl+sqrrstarr);
    real_t vdotbstarstar = ustar*a+vstarstar*bstarstar+wstarstar*cstarstar;
    real_t etotstarstarl = etotstarl-sgnm*sqrrstarl*(vdotbstarl-vdotbstarstar);
    real_t etotstarstarr = etotstarr+sgnm*sqrrstarr*(vdotbstarr-vdotbstarstar);
    
    // sample the solution at x/t=0
    real_t ro, uo, vo, wo, bo, co, ptoto, etoto, vdotbo;
    if(sl>0) { // flow is supersonic, return upwind variables
      ro=rl;
      uo=ul;
      vo=vl;
      wo=wl;
      bo=bl;
      co=cl;
      ptoto=ptotl;
      etoto=etotl;
      vdotbo=vdotbl;
    } else if (sal>0) {
      ro=rstarl;
      uo=ustar;
      vo=vstarl;
      wo=wstarl;
      bo=bstarl;
      co=cstarl;
      ptoto=ptotstar;
      etoto=etotstarl;
      vdotbo=vdotbstarl;
    } else if (ustar>0) {
      ro=rstarl;
      uo=ustar;
      vo=vstarstar;
      wo=wstarstar;
      bo=bstarstar;
      co=cstarstar;
      ptoto=ptotstar;
      etoto=etotstarstarl;
      vdotbo=vdotbstarstar;
    } else if (sar>0) {
      ro=rstarr;
      uo=ustar;
      vo=vstarstar;
      wo=wstarstar;
      bo=bstarstar;
      co=cstarstar;
      ptoto=ptotstar;
      etoto=etotstarstarr;
      vdotbo=vdotbstarstar;
    } else if (sr>0) {
      ro=rstarr;
      uo=ustar;
      vo=vstarr;
      wo=wstarr;
      bo=bstarr;
      co=cstarr;
      ptoto=ptotstar;
      etoto=etotstarr;
      vdotbo=vdotbstarr;
    } else { // flow is supersonic, return upwind variables
      ro=rr;
      uo=ur;
      vo=vr;
      wo=wr;
      bo=br;
      co=cr;
      ptoto=ptotr;
      etoto=etotr;
      vdotbo=vdotbr;
    }
    
    // compute the godunov flux
    flux.d = ro*uo;
    flux.p = (etoto+ptoto)*uo-a*vdotbo;
    flux.u = ro*uo*uo-a*a+ptoto; /* *** WARNING *** : ptoto used here (this is only valid for cartesian geometry) ! */
    flux.v = ro*uo*vo-a*bo;
    flux.w = ro*uo*wo-a*co;
    flux.bx = 0.0;
    flux.by = bo*uo-a*vo;
    flux.bz = co*uo-a*wo;
    
  } // riemann_hlld
      
  /**
   * Compute emf from qEdge state vector via a 2D magnetic Riemann
   * solver (see routine cmp_mag_flux in DUMSES).
   *
   * @param[in] qEdge array containing input states qRT, qLT, qRB, qLB
   * @param[in] xPos x position in space (only needed for shearing box correction terms).
   * @return emf 
   *
   * template parameters:
   *
   * @tparam emfDir plays the role of xdim/lor in DUMSES routine
   * cmp_mag_flx, i.e. define which EMF will be computed (how to define
   * parallel/orthogonal velocity). emfDir identifies the orthogonal direction.
   *
   * \note the global parameter magRiemannSolver is used to choose the
   * 2D magnetic Riemann solver.
   *
   * TODO: make xPos parameter non-optional
   */
  template <EmfDir emfDir>
  KOKKOS_INLINE_FUNCTION
  real_t compute_emf(MHDState qEdge [4], real_t xPos=0) const
  {
  
    // define alias reference to input arrays
    MHDState &qRT = qEdge[IRT];
    MHDState &qLT = qEdge[ILT];
    MHDState &qRB = qEdge[IRB];
    MHDState &qLB = qEdge[ILB];

    // defines alias reference to intermediate state before applying a
    // magnetic Riemann solver
    MHDState qLLRR[4];
    MHDState &qLL = qLLRR[ILL];
    MHDState &qRL = qLLRR[IRL];
    MHDState &qLR = qLLRR[ILR];
    MHDState &qRR = qLLRR[IRR];
  
    // density
    qLL.d = qRT.d;
    qRL.d = qLT.d;
    qLR.d = qRB.d;
    qRR.d = qLB.d;

    // pressure
    // ISOTHERMAL
    real_t cIso = params.settings.cIso;
    if (cIso > 0) {
      qLL.p = qLL.d*cIso*cIso;
      qRL.p = qRL.d*cIso*cIso;
      qLR.p = qLR.d*cIso*cIso;
      qRR.p = qRR.d*cIso*cIso;
    } else {
      qLL.p = qRT.p;
      qRL.p = qLT.p;
      qLR.p = qRB.p;
      qRR.p = qLB.p;
    }

    // iu, iv : parallel velocity indexes
    // iw     : orthogonal velocity index
    // ia, ib, ic : idem for magnetic field
    //int iu, iv, iw, ia, ib, ic;
    if (emfDir == EMFZ) {

      //iu = IU; iv = IV; iw = IW;
      //ia = IA; ib = IB, ic = IC;
      
      // First parallel velocity 
      qLL.u = qRT.u;
      qRL.u = qLT.u;
      qLR.u = qRB.u;
      qRR.u = qLB.u;
      
      // Second parallel velocity 
      qLL.v = qRT.v;
      qRL.v = qLT.v;
      qLR.v = qRB.v;
      qRR.v = qLB.v;
      
      // First parallel magnetic field (enforce continuity)
      qLL.bx = HALF_F * ( qRT.bx + qLT.bx );
      qRL.bx = HALF_F * ( qRT.bx + qLT.bx );
      qLR.bx = HALF_F * ( qRB.bx + qLB.bx );
      qRR.bx = HALF_F * ( qRB.bx + qLB.bx );
      
      // Second parallel magnetic field (enforce continuity)
      qLL.by = HALF_F * ( qRT.by + qRB.by );
      qRL.by = HALF_F * ( qLT.by + qLB.by );
      qLR.by = HALF_F * ( qRT.by + qRB.by );
      qRR.by = HALF_F * ( qLT.by + qLB.by );
      
      // Orthogonal velocity 
      qLL.w = qRT.w;
      qRL.w = qLT.w;
      qLR.w = qRB.w;
      qRR.w = qLB.w;
      
      // Orthogonal magnetic Field
      qLL.bz = qRT.bz;
      qRL.bz = qLT.bz;
      qLR.bz = qRB.bz;
      qRR.bz = qLB.bz;

    } else if (emfDir == EMFY) {

      //iu = IW; iv = IU; iw = IV;
      //ia = IC; ib = IA, ic = IB;
      
      // First parallel velocity 
      qLL.u = qRT.w;
      qRL.u = qLT.w;
      qLR.u = qRB.w;
      qRR.u = qLB.w;
      
      // Second parallel velocity 
      qLL.v = qRT.u;
      qRL.v = qLT.u;
      qLR.v = qRB.u;
      qRR.v = qLB.u;
      
      // First parallel magnetic field (enforce continuity)
      qLL.bx = HALF_F * ( qRT.bz + qLT.bz );
      qRL.bx = HALF_F * ( qRT.bz + qLT.bz );
      qLR.bx = HALF_F * ( qRB.bz + qLB.bz );
      qRR.bx = HALF_F * ( qRB.bz + qLB.bz );
      
      // Second parallel magnetic field (enforce continuity)
      qLL.by = HALF_F * ( qRT.bx + qRB.bx );
      qRL.by = HALF_F * ( qLT.bx + qLB.bx );
      qLR.by = HALF_F * ( qRT.bx + qRB.bx );
      qRR.by = HALF_F * ( qLT.bx + qLB.bx );
      
      // Orthogonal velocity 
      qLL.w = qRT.v;
      qRL.w = qLT.v;
      qLR.w = qRB.v;
      qRR.w = qLB.v;
      
      // Orthogonal magnetic Field
      qLL.bz = qRT.by;
      qRL.bz = qLT.by;
      qLR.bz = qRB.by;
      qRR.bz = qLB.by;
      
    } else { // emfDir == EMFX

      //iu = IV; iv = IW; iw = IU;
      //ia = IB; ib = IC, ic = IA;

      // First parallel velocity 
      qLL.u = qRT.v;
      qRL.u = qLT.v;
      qLR.u = qRB.v;
      qRR.u = qLB.v;
      
      // Second parallel velocity 
      qLL.v = qRT.w;
      qRL.v = qLT.w;
      qLR.v = qRB.w;
      qRR.v = qLB.w;
      
      // First parallel magnetic field (enforce continuity)
      qLL.bx = HALF_F * ( qRT.by + qLT.by );
      qRL.bx = HALF_F * ( qRT.by + qLT.by );
      qLR.bx = HALF_F * ( qRB.by + qLB.by );
      qRR.bx = HALF_F * ( qRB.by + qLB.by );
      
      // Second parallel magnetic field (enforce continuity)
      qLL.by = HALF_F * ( qRT.bz + qRB.bz );
      qRL.by = HALF_F * ( qLT.bz + qLB.bz );
      qLR.by = HALF_F * ( qRT.bz + qRB.bz );
      qRR.by = HALF_F * ( qLT.bz + qLB.bz );
      
      // Orthogonal velocity 
      qLL.w = qRT.u;
      qRL.w = qLT.u;
      qLR.w = qRB.u;
      qRR.w = qLB.u;
      
      // Orthogonal magnetic Field
      qLL.bz = qRT.bx;
      qRL.bz = qLT.bx;
      qLR.bz = qRB.bx;
      qRR.bz = qLB.bx;
    }

  
    // Compute final fluxes
  
    // vx*by - vy*bx at the four edge centers
    real_t eLLRR[4];
    real_t &ELL = eLLRR[ILL];
    real_t &ERL = eLLRR[IRL];
    real_t &ELR = eLLRR[ILR];
    real_t &ERR = eLLRR[IRR];

    ELL = qLL.u*qLL.by - qLL.v*qLL.bx;
    ERL = qRL.u*qRL.by - qRL.v*qRL.bx;
    ELR = qLR.u*qLR.by - qLR.v*qLR.bx;
    ERR = qRR.u*qRR.by - qRR.v*qRR.bx;

    real_t emf=0;
    // mag_riemann2d<>
    //if (params.magRiemannSolver == MAG_HLLD) {
    emf = mag_riemann2d_hlld(qLLRR, eLLRR);
    // } else if (params.magRiemannSolver == MAG_HLLA) {
    //   emf = mag_riemann2d_hlla(qLLRR, eLLRR);
    // } else if (params.magRiemannSolver == MAG_HLLF) {
    //   emf = mag_riemann2d_hllf(qLLRR, eLLRR);
    // } else if (params.magRiemannSolver == MAG_LLF) {
    //   emf = mag_riemann2d_llf(qLLRR, eLLRR);
    // }

    /* upwind solver in case of the shearing box */
    // if ( /* cartesian */ (params.settings.Omega0>0) /* and not fargo */ ) {
    //   if (emfDir==EMFX) {
    // 	real_t shear = -1.5 * params.Omega0 * xPos;
    // 	if (shear>0) {
    // 	  emf += shear * qLL.by;
    // 	} else {
    // 	  emf += shear * qRR.by;
    // 	}
    //   }
    //   if (emfDir==EMFZ) {
    // 	real_t shear = -1.5 * params.Omega0 * (xPos - params.dx/2);
    // 	if (shear>0) {
    // 	  emf -= shear * qLL.bx;
    // 	} else {
    // 	  emf -= shear * qRR.bx;
    // 	}
    //   }
    // }

    return emf;

  } // compute_emf

  /**
   * 2D magnetic riemann solver of type HLLD
   *
   */
  KOKKOS_INLINE_FUNCTION
  real_t mag_riemann2d_hlld(MHDState qLLRR[4],
			    real_t eLLRR[4]) const
  {

    // alias reference to input arrays
    MHDState &qLL = qLLRR[ILL];
    MHDState &qRL = qLLRR[IRL];
    MHDState &qLR = qLLRR[ILR];
    MHDState &qRR = qLLRR[IRR];

    real_t &ELL = eLLRR[ILL];
    real_t &ERL = eLLRR[IRL];
    real_t &ELR = eLLRR[ILR];
    real_t &ERR = eLLRR[IRR];
    //real_t ELL,ERL,ELR,ERR;

    real_t &rLL=qLL.d; real_t &pLL=qLL.p; 
    real_t &uLL=qLL.u; real_t &vLL=qLL.v; 
    real_t &aLL=qLL.bx; real_t &bLL=qLL.by ; real_t &cLL=qLL.bz;
  
    real_t &rLR=qLR.d; real_t &pLR=qLR.p; 
    real_t &uLR=qLR.u; real_t &vLR=qLR.v; 
    real_t &aLR=qLR.bx; real_t &bLR=qLR.by ; real_t &cLR=qLR.bz;
  
    real_t &rRL=qRL.d; real_t &pRL=qRL.p; 
    real_t &uRL=qRL.u; real_t &vRL=qRL.v; 
    real_t &aRL=qRL.bx; real_t &bRL=qRL.by ; real_t &cRL=qRL.bz;

    real_t &rRR=qRR.d; real_t &pRR=qRR.p; 
    real_t &uRR=qRR.u; real_t &vRR=qRR.v; 
    real_t &aRR=qRR.bx; real_t &bRR=qRR.by ; real_t &cRR=qRR.bz;
  
    // Compute 4 fast magnetosonic velocity relative to x direction
    real_t cFastLLx = find_speed_fast<IX>(qLL);
    real_t cFastLRx = find_speed_fast<IX>(qLR);
    real_t cFastRLx = find_speed_fast<IX>(qRL);
    real_t cFastRRx = find_speed_fast<IX>(qRR);

    // Compute 4 fast magnetosonic velocity relative to y direction 
    real_t cFastLLy = find_speed_fast<IY>(qLL);
    real_t cFastLRy = find_speed_fast<IY>(qLR);
    real_t cFastRLy = find_speed_fast<IY>(qRL);
    real_t cFastRRy = find_speed_fast<IY>(qRR);
  
    // TODO : write a find_speed that computes the 2 speeds together (in
    // a single routine -> factorize computation of cFastLLx and cFastLLy

    real_t SL = FMIN4(uLL,uLR,uRL,uRR) - FMAX4(cFastLLx,cFastLRx,cFastRLx,cFastRRx);
    real_t SR = FMAX4(uLL,uLR,uRL,uRR) + FMAX4(cFastLLx,cFastLRx,cFastRLx,cFastRRx);
    real_t SB = FMIN4(vLL,vLR,vRL,vRR) - FMAX4(cFastLLy,cFastLRy,cFastRLy,cFastRRy);
    real_t ST = FMAX4(vLL,vLR,vRL,vRR) + FMAX4(cFastLLy,cFastLRy,cFastRLy,cFastRRy);

    /*ELL = uLL*bLL - vLL*aLL;
      ELR = uLR*bLR - vLR*aLR;
      ERL = uRL*bRL - vRL*aRL;
      ERR = uRR*bRR - vRR*aRR;*/
  
    real_t PtotLL = pLL + HALF_F * (aLL*aLL + bLL*bLL + cLL*cLL);
    real_t PtotLR = pLR + HALF_F * (aLR*aLR + bLR*bLR + cLR*cLR);
    real_t PtotRL = pRL + HALF_F * (aRL*aRL + bRL*bRL + cRL*cRL);
    real_t PtotRR = pRR + HALF_F * (aRR*aRR + bRR*bRR + cRR*cRR);
  
    real_t rcLLx = rLL * (uLL-SL); real_t rcRLx = rRL *(SR-uRL);
    real_t rcLRx = rLR * (uLR-SL); real_t rcRRx = rRR *(SR-uRR);
    real_t rcLLy = rLL * (vLL-SB); real_t rcLRy = rLR *(ST-vLR);
    real_t rcRLy = rRL * (vRL-SB); real_t rcRRy = rRR *(ST-vRR);

    real_t ustar = (rcLLx*uLL + rcLRx*uLR + rcRLx*uRL + rcRRx*uRR +
		    (PtotLL - PtotRL + PtotLR - PtotRR) ) / (rcLLx + rcLRx + 
							     rcRLx + rcRRx);
    real_t vstar = (rcLLy*vLL + rcLRy*vLR + rcRLy*vRL + rcRRy*vRR +
		    (PtotLL - PtotLR + PtotRL - PtotRR) ) / (rcLLy + rcLRy + 
							     rcRLy + rcRRy);
  
    real_t rstarLLx = rLL * (SL-uLL) / (SL-ustar);
    real_t BstarLL  = bLL * (SL-uLL) / (SL-ustar);
    real_t rstarLLy = rLL * (SB-vLL) / (SB-vstar); 
    real_t AstarLL  = aLL * (SB-vLL) / (SB-vstar);
    real_t rstarLL  = rLL * (SL-uLL) / (SL-ustar) 
      *                     (SB-vLL) / (SB-vstar);
    real_t EstarLLx = ustar * BstarLL - vLL   * aLL;
    real_t EstarLLy = uLL   * bLL     - vstar * AstarLL;
    real_t EstarLL  = ustar * BstarLL - vstar * AstarLL;
  
    real_t rstarLRx = rLR * (SL-uLR) / (SL-ustar); 
    real_t BstarLR  = bLR * (SL-uLR) / (SL-ustar);
    real_t rstarLRy = rLR * (ST-vLR) / (ST-vstar); 
    real_t AstarLR  = aLR * (ST-vLR) / (ST-vstar);
    real_t rstarLR  = rLR * (SL-uLR) / (SL-ustar) * (ST-vLR) / (ST-vstar);
    real_t EstarLRx = ustar * BstarLR - vLR   * aLR;
    real_t EstarLRy = uLR   * bLR     - vstar * AstarLR;
    real_t EstarLR  = ustar * BstarLR - vstar * AstarLR;

    real_t rstarRLx = rRL * (SR-uRL) / (SR-ustar); 
    real_t BstarRL  = bRL * (SR-uRL) / (SR-ustar);
    real_t rstarRLy = rRL * (SB-vRL) / (SB-vstar); 
    real_t AstarRL  = aRL * (SB-vRL) / (SB-vstar);
    real_t rstarRL  = rRL * (SR-uRL) / (SR-ustar) * (SB-vRL) / (SB-vstar);
    real_t EstarRLx = ustar * BstarRL - vRL   * aRL;
    real_t EstarRLy = uRL   * bRL     - vstar * AstarRL;
    real_t EstarRL  = ustar * BstarRL - vstar * AstarRL;
  
    real_t rstarRRx = rRR * (SR-uRR) / (SR-ustar); 
    real_t BstarRR  = bRR * (SR-uRR) / (SR-ustar);
    real_t rstarRRy = rRR * (ST-vRR) / (ST-vstar); 
    real_t AstarRR  = aRR * (ST-vRR) / (ST-vstar);
    real_t rstarRR  = rRR * (SR-uRR) / (SR-ustar) * (ST-vRR) / (ST-vstar);
    real_t EstarRRx = ustar * BstarRR - vRR   * aRR;
    real_t EstarRRy = uRR   * bRR     - vstar * AstarRR;
    real_t EstarRR  = ustar * BstarRR - vstar * AstarRR;

    real_t calfvenL = FMAX5(FABS(aLR)/SQRT(rstarLRx), FABS(AstarLR)/SQRT(rstarLR), 
			    FABS(aLL)/SQRT(rstarLLx), FABS(AstarLL)/SQRT(rstarLL), 
			    params.settings.smallc);
    real_t calfvenR = FMAX5(FABS(aRR)/SQRT(rstarRRx), FABS(AstarRR)/SQRT(rstarRR),
			    FABS(aRL)/SQRT(rstarRLx), FABS(AstarRL)/SQRT(rstarRL), 
			    params.settings.smallc);
    real_t calfvenB = FMAX5(FABS(bLL)/SQRT(rstarLLy), FABS(BstarLL)/SQRT(rstarLL), 
			    FABS(bRL)/SQRT(rstarRLy), FABS(BstarRL)/SQRT(rstarRL), 
			    params.settings.smallc);
    real_t calfvenT = FMAX5(FABS(bLR)/SQRT(rstarLRy), FABS(BstarLR)/SQRT(rstarLR), 
			    FABS(bRR)/SQRT(rstarRRy), FABS(BstarRR)/SQRT(rstarRR), 
			    params.settings.smallc);

    real_t SAL = FMIN(ustar - calfvenL, (real_t) ZERO_F); 
    real_t SAR = FMAX(ustar + calfvenR, (real_t) ZERO_F);
    real_t SAB = FMIN(vstar - calfvenB, (real_t) ZERO_F); 
    real_t SAT = FMAX(vstar + calfvenT, (real_t) ZERO_F);

    real_t AstarT = (SAR*AstarRR - SAL*AstarLR) / (SAR-SAL); 
    real_t AstarB = (SAR*AstarRL - SAL*AstarLL) / (SAR-SAL);
  
    real_t BstarR = (SAT*BstarRR - SAB*BstarRL) / (SAT-SAB); 
    real_t BstarL = (SAT*BstarLR - SAB*BstarLL) / (SAT-SAB);

    // finally get emf E
    real_t E=0, tmpE=0;

    // the following part is slightly different from the original fortran
    // code since it has to much different branches
    // which generate to much branch divergence in CUDA !!!

    // compute sort of boolean (don't know if signbit is available)
    int SB_pos = (int) (1+COPYSIGN(ONE_F,SB))/2, SB_neg = 1-SB_pos;
    int ST_pos = (int) (1+COPYSIGN(ONE_F,ST))/2, ST_neg = 1-ST_pos;
    int SL_pos = (int) (1+COPYSIGN(ONE_F,SL))/2, SL_neg = 1-SL_pos;
    int SR_pos = (int) (1+COPYSIGN(ONE_F,SR))/2, SR_neg = 1-SR_pos;

    // else
    tmpE = (SAL*SAB*EstarRR-SAL*SAT*EstarRL - 
	    SAR*SAB*EstarLR+SAR*SAT*EstarLL)/(SAR-SAL)/(SAT-SAB) - 
      SAT*SAB/(SAT-SAB)*(AstarT-AstarB) + 
      SAR*SAL/(SAR-SAL)*(BstarR-BstarL);
    E += (SB_neg * ST_pos * SL_neg * SR_pos) * tmpE;

    // SB>0
    tmpE = (SAR*EstarLLx-SAL*EstarRLx+SAR*SAL*(bRL-bLL))/(SAR-SAL);
    tmpE = SL_pos*ELL + SL_neg*SR_neg*ERL + SL_neg*SR_pos*tmpE;
    E += SB_pos * tmpE;

    // ST<0
    tmpE = (SAR*EstarLRx-SAL*EstarRRx+SAR*SAL*(bRR-bLR))/(SAR-SAL);
    tmpE = SL_pos*ELR + SL_neg*SR_neg*ERR + SL_neg*SR_pos*tmpE;
    E += (SB_neg * ST_neg) * tmpE;

    // SL>0
    tmpE = (SAT*EstarLLy-SAB*EstarLRy-SAT*SAB*(aLR-aLL))/(SAT-SAB);
    E += (SB_neg * ST_pos * SL_pos) * tmpE;

    // SR<0
    tmpE = (SAT*EstarRLy-SAB*EstarRRy-SAT*SAB*(aRR-aRL))/(SAT-SAB);
    E += (SB_neg * ST_pos * SL_neg * SR_neg) * tmpE;


    /*
      if(SB>ZERO_F) {
      if(SL>ZERO_F) {
      E=ELL;
      } else if(SR<ZERO_F) {
      E=ERL;
      } else {
      E=(SAR*EstarLLx-SAL*EstarRLx+SAR*SAL*(bRL-bLL))/(SAR-SAL);
      }
      } else if (ST<ZERO_F) {
      if(SL>ZERO_F) {
      E=ELR;
      } else if(SR<ZERO_F) {
      E=ERR;
      } else {
      E=(SAR*EstarLRx-SAL*EstarRRx+SAR*SAL*(bRR-bLR))/(SAR-SAL);
      }
      } else if (SL>ZERO_F) {
      E=(SAT*EstarLLy-SAB*EstarLRy-SAT*SAB*(aLR-aLL))/(SAT-SAB);
      } else if (SR<ZERO_F) {
      E=(SAT*EstarRLy-SAB*EstarRRy-SAT*SAB*(aRR-aRL))/(SAT-SAB);
      } else {
      E = (SAL*SAB*EstarRR-SAL*SAT*EstarRL - 
      SAR*SAB*EstarLR+SAR*SAT*EstarLL)/(SAR-SAL)/(SAT-SAB) - 
      SAT*SAB/(SAT-SAB)*(AstarT-AstarB) + 
      SAR*SAL/(SAR-SAL)*(BstarR-BstarL);
      }
    */

    return E;

  } // mag_riemann2d_hlld

}; // class MHDBaseFunctor2D

} // namespace muscl

} // namespace ppkMHD

#endif // MHD_BASE_FUNCTOR_2D_H_
