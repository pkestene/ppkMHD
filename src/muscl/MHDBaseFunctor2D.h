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

    q[ID]  = data(i,j, ID);
    q[IP]  = data(i,j, IP);
    q[IU]  = data(i,j, IU);
    q[IV]  = data(i,j, IV);
    q[IW]  = data(i,j, IW);
    q[IBX] = data(i,j, IBX);
    q[IBY] = data(i,j, IBY);
    q[IBZ] = data(i,j, IBZ);
    
  } // get_state

  /**
   * Copy q into data(i,j).
   */
  KOKKOS_INLINE_FUNCTION
  void set_state(DataArray data, int i, int j, const MHDState& q) const
  {

    data(i,j, ID)  = q[ID];
    data(i,j, IP)  = q[IP];
    data(i,j, IU)  = q[IU];
    data(i,j, IV)  = q[IV];
    data(i,j, IW)  = q[IW];
    data(i,j, IBX) = q[IBX];
    data(i,j, IBY) = q[IBY];
    data(i,j, IBZ) = q[IBZ];
    
  } // set_state

  /**
   *
   */
  KOKKOS_INLINE_FUNCTION
  void get_magField(const DataArray& data, int i, int j, BField& b) const
  {

    b[IBFX] = data(i,j, IBX);
    b[IBFY] = data(i,j, IBY);
    b[IBFZ] = data(i,j, IBZ);
    
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
    q[ID] = fmax(u[ID], smallr);

    // compute velocities
    q[IU] = u[IU] / q[ID];
    q[IV] = u[IV] / q[ID];
    q[IW] = u[IW] / q[ID];

    // compute cell-centered magnetic field
    q[IBX] = 0.5 * ( u[IBX] + magFieldNeighbors[0] );
    q[IBY] = 0.5 * ( u[IBY] + magFieldNeighbors[1] );
    q[IBZ] = 0.5 * ( u[IBZ] + magFieldNeighbors[2] );

    // compute specific kinetic energy and magnetic energy
    real_t eken = 0.5 * (q[IU] *q[IU]  + q[IV] *q[IV]  + q[IW] *q[IW] );
    real_t emag = 0.5 * (q[IBX]*q[IBX] + q[IBY]*q[IBY] + q[IBZ]*q[IBZ]);

    // compute pressure

    if (params.settings.cIso > 0) { // isothermal
      
      q[IP] = q[ID] * (params.settings.cIso) * (params.settings.cIso);
      c     =  params.settings.cIso;
      
    } else {
      
      real_t eint = (u[IP] - emag) / q[ID] - eken;

      q[IP] = fmax((params.settings.gamma0-1.0) * q[ID] * eint,
		 q[ID] * params.settings.smallp);
  
      // if (q[IP] < 0) {
      // 	printf("MHD pressure neg !!!\n");
      // }

      // compute speed of sound (should be removed as it is useless, hydro
      // legacy)
      c = sqrt(params.settings.gamma0 * q[IP] / q[ID]);
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
  void slope_unsplit_hydro_2d(const MHDState (&qNb)[3][3],
			      MHDState (&dq)[2]) const
  {			
    real_t slope_type = params.settings.slope_type;

    // index of current cell in the neighborhood
    enum {CENTER=1};

    // aliases to input qState neighbors
    const MHDState &q       = qNb[CENTER  ][CENTER  ];
    const MHDState &qPlusX  = qNb[CENTER+1][CENTER  ];
    const MHDState &qMinusX = qNb[CENTER-1][CENTER  ];
    const MHDState &qPlusY  = qNb[CENTER  ][CENTER+1]; 
    const MHDState &qMinusY = qNb[CENTER  ][CENTER-1];

    MHDState &dqX = dq[IX];
    MHDState &dqY = dq[IY];
 
    if (slope_type==1 or
	slope_type==2) {  // minmod or average

      slope_unsplit_hydro_2d_scalar(q[ID],qPlusX[ID],qMinusX[ID],qPlusY[ID],qMinusY[ID], &(dqX[ID]), &(dqY[ID]));
      slope_unsplit_hydro_2d_scalar(q[IP],qPlusX[IP],qMinusX[IP],qPlusY[IP],qMinusY[IP], &(dqX[IP]), &(dqY[IP]));
      slope_unsplit_hydro_2d_scalar(q[IU],qPlusX[IU],qMinusX[IU],qPlusY[IU],qMinusY[IU], &(dqX[IU]), &(dqY[IU]));
      slope_unsplit_hydro_2d_scalar(q[IV],qPlusX[IV],qMinusX[IV],qPlusY[IV],qMinusY[IV], &(dqX[IV]), &(dqY[IV]));
      slope_unsplit_hydro_2d_scalar(q[IW],qPlusX[IW],qMinusX[IW],qPlusY[IW],qMinusY[IW], &(dqX[IW]), &(dqY[IW]));
      slope_unsplit_hydro_2d_scalar(q[IBX],qPlusX[IBX],qMinusX[IBX],qPlusY[IBX],qMinusY[IBX], &(dqX[IBX]), &(dqY[IBX]));
      slope_unsplit_hydro_2d_scalar(q[IBY],qPlusX[IBY],qMinusX[IBY],qPlusY[IBY],qMinusY[IBY], &(dqX[IBY]), &(dqY[IBY]));
      slope_unsplit_hydro_2d_scalar(q[IBZ],qPlusX[IBZ],qMinusX[IBZ],qPlusY[IBZ],qMinusY[IBZ], &(dqX[IBZ]), &(dqY[IBZ]));
      
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
  void slope_unsplit_mhd_2d(const real_t (&bfNeighbors)[6],
			    real_t (&dbf)[2][3]) const
  {			
    /* layout for face centered magnetic field */
    const real_t &bfx        = bfNeighbors[0];
    const real_t &bfx_yplus  = bfNeighbors[1];
    const real_t &bfx_yminus = bfNeighbors[2];
    const real_t &bfy        = bfNeighbors[3];
    const real_t &bfy_xplus  = bfNeighbors[4];
    const real_t &bfy_xminus = bfNeighbors[5];
  
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
  void trace_unsplit_2d_along_dir(const MHDState& q, 
				  const MHDState& dqX,
				  const MHDState& dqY,
				  real_t dtdx, 
				  real_t dtdy, 
				  int    faceId,
				  MHDState& qface) const
  {
  
    real_t gamma0 = params.settings.gamma0;
    real_t smallr = params.settings.smallr;

    // Cell centered values
    real_t r =  q[ID];
    real_t p =  q[IP];
    real_t u =  q[IU];
    real_t v =  q[IV];
  
    // TVD slopes in all directions
    real_t drx = dqX[ID];
    real_t dpx = dqX[IP];
    real_t dux = dqX[IU];
    real_t dvx = dqX[IV];
  
    real_t dry = dqY[ID];
    real_t dpy = dqY[IP];
    real_t duy = dqY[IU];
    real_t dvy = dqY[IV];
  
    // source terms (with transverse derivatives)
    real_t sr0 = -u*drx-v*dry - (dux+dvy)*r;
    real_t sp0 = -u*dpx-v*dpy - (dux+dvy)*gamma0*p;
    real_t su0 = -u*dux-v*duy - (dpx    )/r;
    real_t sv0 = -u*dvx-v*dvy - (dpy    )/r;
  
    if (faceId == FACE_XMIN) {
      // Right state at left interface
      qface[ID] = r - 0.5*drx + sr0*dtdx*0.5;
      qface[IP] = p - 0.5*dpx + sp0*dtdx*0.5;
      qface[IU] = u - 0.5*dux + su0*dtdx*0.5;
      qface[IV] = v - 0.5*dvx + sv0*dtdx*0.5;
      qface[ID] = fmax(smallr, qface[ID]);
    }

    if (faceId == FACE_XMAX) {
      // Left state at right interface
      qface[ID] = r + 0.5*drx + sr0*dtdx*0.5;
      qface[IP] = p + 0.5*dpx + sp0*dtdx*0.5;
      qface[IU] = u + 0.5*dux + su0*dtdx*0.5;
      qface[IV] = v + 0.5*dvx + sv0*dtdx*0.5;
      qface[ID] = fmax(smallr, qface[ID]);
    }
  
    if (faceId == FACE_YMIN) {
      // Top state at bottom interface
      qface[ID] = r - 0.5*dry + sr0*dtdy*0.5;
      qface[IP] = p - 0.5*dpy + sp0*dtdy*0.5;
      qface[IU] = u - 0.5*duy + su0*dtdy*0.5;
      qface[IV] = v - 0.5*dvy + sv0*dtdy*0.5;
      qface[ID] = fmax(smallr, qface[ID]);
    }

    if (faceId == FACE_YMAX) {
      // Bottom state at top interface
      qface[ID] = r + 0.5*dry + sr0*dtdy*0.5;
      qface[IP] = p + 0.5*dpy + sp0*dtdy*0.5;
      qface[IU] = u + 0.5*duy + su0*dtdy*0.5;
      qface[IV] = v + 0.5*dvy + sv0*dtdy*0.5;
      qface[ID] = fmax(smallr, qface[ID]);
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
  void trace_unsplit_hydro_2d(const MHDState& q,
			      const MHDState& dqX,
			      const MHDState& dqY,
			      real_t dtdx,
			      real_t dtdy,
			      MHDState& qm_x,
			      MHDState& qm_y,
			      MHDState& qp_x,
			      MHDState& qp_y) const
  {
  
    real_t gamma0 = params.settings.gamma0;
    real_t smallr = params.settings.smallr;
    real_t smallp = params.settings.smallp;

    // Cell centered values
    real_t r = q[ID];
    real_t p = q[IP];
    real_t u = q[IU];
    real_t v = q[IV];

    // Cell centered TVD slopes in X direction
    real_t drx = dqX[ID];  drx *= 0.5;
    real_t dpx = dqX[IP];  dpx *= 0.5;
    real_t dux = dqX[IU];  dux *= 0.5;
    real_t dvx = dqX[IV];  dvx *= 0.5;
  
    // Cell centered TVD slopes in Y direction
    real_t dry = dqY[ID];  dry *= 0.5;
    real_t dpy = dqY[IP];  dpy *= 0.5;
    real_t duy = dqY[IU];  duy *= 0.5;
    real_t dvy = dqY[IV];  dvy *= 0.5;

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
    qp_x[ID] = r - drx;
    qp_x[IU] = u - dux;
    qp_x[IV] = v - dvx;
    qp_x[IP] = p - dpx;
    qp_x[ID] = fmax(smallr,  qp_x[ID]);
    qp_x[IP] = fmax(smallp * qp_x[ID], qp_x[IP]);
  
    // Face averaged left state at right interface
    qm_x[ID] = r + drx;
    qm_x[IU] = u + dux;
    qm_x[IV] = v + dvx;
    qm_x[IP] = p + dpx;
    qm_x[ID] = fmax(smallr,  qm_x[ID]);
    qm_x[IP] = fmax(smallp * qm_x[ID], qm_x[IP]);

    // Face averaged top state at bottom interface
    qp_y[ID] = r - dry;
    qp_y[IU] = u - duy;
    qp_y[IV] = v - dvy;
    qp_y[IP] = p - dpy;
    qp_y[ID] = fmax(smallr,  qp_y[ID]);
    qp_y[IP] = fmax(smallp * qp_y[ID], qp_y[IP]);
  
    // Face averaged bottom state at top interface
    qm_y[ID] = r + dry;
    qm_y[IU] = u + duy;
    qm_y[IV] = v + dvy;
    qm_y[IP] = p + dpy;
    qm_y[ID] = fmax(smallr,  qm_y[ID]);
    qm_y[IP] = fmax(smallp * qm_y[ID], qm_y[IP]);
  
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
  void trace_unsplit_mhd_2d(const MHDState (&qNb)[3][3],
			    const BField (&bfNb)[4][4],
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

    const MHDState &q = qNb[CENTER][CENTER]; // current cell (neighborhood center)

    // compute u,v,A,B,Ez (electric field)
    real_t Ez[2][2];
    for (int di=0; di<2; di++)
      for (int dj=0; dj<2; dj++) {
      
	int centerX = CENTER+di;
	int centerY = CENTER+dj;
	real_t u  = 0.25f *  (qNb[centerX-1][centerY-1][IU] + 
			      qNb[centerX-1][centerY  ][IU] + 
			      qNb[centerX  ][centerY-1][IU] + 
			      qNb[centerX  ][centerY  ][IU]); 
      
	real_t v  = 0.25f *  (qNb[centerX-1][centerY-1][IV] +
			      qNb[centerX-1][centerY  ][IV] +
			      qNb[centerX  ][centerY-1][IV] + 
			      qNb[centerX  ][centerY  ][IV]);
      
	real_t A  = 0.5f  * (bfNb[centerX  ][centerY-1][IBFX] + 
			     bfNb[centerX  ][centerY  ][IBFX]);

	real_t B  = 0.5f  * (bfNb[centerX-1][centerY  ][IBFY] + 
			     bfNb[centerX  ][centerY  ][IBFY]);
      
	Ez[di][dj] = u*B-v*A;
      }

    // Electric field
    real_t &ELL = Ez[0][0];
    real_t &ELR = Ez[0][1];
    real_t &ERL = Ez[1][0];
    real_t &ERR = Ez[1][1];

    // Cell centered values
    real_t r = q[ID];
    real_t p = q[IP];
    real_t u = q[IU];
    real_t v = q[IV];
    real_t w = q[IW];            
    real_t A = q[IBX];
    real_t B = q[IBY];
    real_t C = q[IBZ];            
    
    // Face centered variables
    real_t AL =  bfNb[CENTER  ][CENTER  ][IBFX];
    real_t AR =  bfNb[CENTER+1][CENTER  ][IBFX];
    real_t BL =  bfNb[CENTER  ][CENTER  ][IBFY];
    real_t BR =  bfNb[CENTER  ][CENTER+1][IBFY];

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
    real_t drx = dq[IX][ID];  drx *= 0.5;
    real_t dpx = dq[IX][IP];  dpx *= 0.5;
    real_t dux = dq[IX][IU];  dux *= 0.5;
    real_t dvx = dq[IX][IV];  dvx *= 0.5;
    real_t dwx = dq[IX][IW];  dwx *= 0.5;
    real_t dCx = dq[IX][IBZ];  dCx *= 0.5;
    real_t dBx = dq[IX][IBY];  dBx *= 0.5;
  
    // Cell centered TVD slopes in Y direction
    real_t dry = dq[IY][ID];  dry *= 0.5;
    real_t dpy = dq[IY][IP];  dpy *= 0.5;
    real_t duy = dq[IY][IU];  duy *= 0.5;
    real_t dvy = dq[IY][IV];  dvy *= 0.5;
    real_t dwy = dq[IY][IW];  dwy *= 0.5;
    real_t dCy = dq[IY][IBZ];  dCy *= 0.5;
    real_t dAy = dq[IY][IBX];  dAy *= 0.5;
  
    /*
     * compute dbf slopes needed for Face centered TVD slopes in transverse direction
     */
    real_t bfNeighbors[6];
    real_t dbf[2][3];
    real_t (&dbfX)[3] = dbf[IX];
    real_t (&dbfY)[3] = dbf[IY];
  
    bfNeighbors[0] =  bfNb[CENTER  ][CENTER  ][IBFX];
    bfNeighbors[1] =  bfNb[CENTER  ][CENTER+1][IBFX];
    bfNeighbors[2] =  bfNb[CENTER  ][CENTER-1][IBFX];
    bfNeighbors[3] =  bfNb[CENTER  ][CENTER  ][IBFY];
    bfNeighbors[4] =  bfNb[CENTER+1][CENTER  ][IBFY];
    bfNeighbors[5] =  bfNb[CENTER-1][CENTER  ][IBFY];
  
    slope_unsplit_mhd_2d(bfNeighbors, dbf);
  
    // Face centered TVD slopes in transverse direction
    real_t dALy = 0.5 * dbfY[IX];
    real_t dBLx = 0.5 * dbfX[IY];

    // change neighbors to i+1, j and recompute dbf
    bfNeighbors[0] =  bfNb[CENTER+1][CENTER  ][IBFX];
    bfNeighbors[1] =  bfNb[CENTER+1][CENTER+1][IBFX];
    bfNeighbors[2] =  bfNb[CENTER+1][CENTER-1][IBFX];
    bfNeighbors[3] =  bfNb[CENTER+1][CENTER  ][IBFY];
    bfNeighbors[4] =  bfNb[CENTER+2][CENTER  ][IBFY];
    bfNeighbors[5] =  bfNb[CENTER  ][CENTER  ][IBFY];

    slope_unsplit_mhd_2d(bfNeighbors, dbf);  

    real_t dARy = 0.5 * dbfY[IX];

    // change neighbors to i, j+1 and recompute dbf
    bfNeighbors[0] =  bfNb[CENTER  ][CENTER+1][IBFX];
    bfNeighbors[1] =  bfNb[CENTER  ][CENTER+2][IBFX];
    bfNeighbors[2] =  bfNb[CENTER  ][CENTER  ][IBFX];
    bfNeighbors[3] =  bfNb[CENTER  ][CENTER+1][IBFY];
    bfNeighbors[4] =  bfNb[CENTER+1][CENTER+1][IBFY];
    bfNeighbors[5] =  bfNb[CENTER-1][CENTER+1][IBFY];

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
    qp[0][ID] = r - drx;
    qp[0][IU] = u - dux;
    qp[0][IV] = v - dvx;
    qp[0][IW] = w - dwx;
    qp[0][IP] = p - dpx;
    qp[0][IBX] = AL;
    qp[0][IBY] = B - dBx;
    qp[0][IBZ] = C - dCx;
    qp[0][ID] = FMAX(smallR, qp[0][ID]);
    qp[0][IP] = FMAX(smallp*qp[0][ID], qp[0][IP]);
  
    // Left state at right interface
    qm[0][ID] = r + drx;
    qm[0][IU] = u + dux;
    qm[0][IV] = v + dvx;
    qm[0][IW] = w + dwx;
    qm[0][IP] = p + dpx;
    qm[0][IBX] = AR;
    qm[0][IBY] = B + dBx;
    qm[0][IBZ] = C + dCx;
    qm[0][ID] = FMAX(smallR, qm[0][ID]);
    qm[0][IP] = FMAX(smallp*qm[0][ID], qm[0][IP]);
  
    // Top state at bottom interface
    qp[1][ID] = r - dry;
    qp[1][IU] = u - duy;
    qp[1][IV] = v - dvy;
    qp[1][IW] = w - dwy;
    qp[1][IP] = p - dpy;
    qp[1][IBX] = A - dAy;
    qp[1][IBY] = BL;
    qp[1][IBZ] = C - dCy;
    qp[1][ID] = FMAX(smallR, qp[1][ID]);
    qp[1][IP] = FMAX(smallp*qp[1][ID], qp[1][IP]);
  
    // Bottom state at top interface
    qm[1][ID] = r + dry;
    qm[1][IU] = u + duy;
    qm[1][IV] = v + dvy;
    qm[1][IW] = w + dwy;
    qm[1][IP] = p + dpy;
    qm[1][IBX] = A + dAy;
    qm[1][IBY] = BR;
    qm[1][IBZ] = C + dCy;
    qm[1][ID] = FMAX(smallR, qm[1][ID]);
    qm[1][IP] = FMAX(smallp*qm[1][ID], qm[1][IP]);
  
  
    // Right-top state (RT->LL)
    qRT[ID] = r + (+drx+dry);
    qRT[IU] = u + (+dux+duy);
    qRT[IV] = v + (+dvx+dvy);
    qRT[IW] = w + (+dwx+dwy);
    qRT[IP] = p + (+dpx+dpy);
    qRT[IBX] = AR+ (   +dARy);
    qRT[IBY] = BR+ (+dBRx   );
    qRT[IBZ] = C + (+dCx+dCy);
    qRT[ID] = FMAX(smallR, qRT[ID]);
    qRT[IP] = FMAX(smallp*qRT[ID], qRT[IP]);
    
    // Right-Bottom state (RB->LR)
    qRB[ID] = r + (+drx-dry);
    qRB[IU] = u + (+dux-duy);
    qRB[IV] = v + (+dvx-dvy);
    qRB[IW] = w + (+dwx-dwy);
    qRB[IP] = p + (+dpx-dpy);
    qRB[IBX] = AR+ (   -dARy);
    qRB[IBY] = BL+ (+dBLx   );
    qRB[IBZ] = C + (+dCx-dCy);
    qRB[ID] = FMAX(smallR, qRB[ID]);
    qRB[IP] = FMAX(smallp*qRB[ID], qRB[IP]);
    
    // Left-Bottom state (LB->RR)
    qLB[ID] = r + (-drx-dry);
    qLB[IU] = u + (-dux-duy);
    qLB[IV] = v + (-dvx-dvy);
    qLB[IW] = w + (-dwx-dwy);
    qLB[IP] = p + (-dpx-dpy);
    qLB[IBX] = AL+ (   -dALy);
    qLB[IBY] = BL+ (-dBLx   );
    qLB[IBZ] = C + (-dCx-dCy);
    qLB[ID] = FMAX(smallR, qLB[ID]);
    qLB[IP] = FMAX(smallp*qLB[ID], qLB[IP]);
    
    // Left-Top state (LT->RL)
    qLT[ID] = r + (-drx+dry);
    qLT[IU] = u + (-dux+duy);
    qLT[IV] = v + (-dvx+dvy);
    qLT[IW] = w + (-dwx+dwy);
    qLT[IP] = p + (-dpx+dpy);
    qLT[IBX] = AL+ (   +dALy);
    qLT[IBY] = BR+ (-dBRx   );
    qLT[IBZ] = C + (-dCx+dCy);
    qLT[ID] = FMAX(smallR, qLT[ID]);
    qLT[IP] = FMAX(smallp*qLT[ID], qLT[IP]);

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
  void slope_unsplit_hydro_2d(const MHDState& q, 
			      const MHDState& qPlusX, 
			      const MHDState& qMinusX,
			      const MHDState& qPlusY,
			      const MHDState& qMinusY,
			      MHDState& dqX,
			      MHDState& dqY) const
  {
  
    real_t slope_type = params.settings.slope_type;

    if (slope_type==0) {

      dqX[ID] = ZERO_F;
      dqX[IP] = ZERO_F;
      dqX[IU] = ZERO_F;
      dqX[IV] = ZERO_F;

      dqY[ID] = ZERO_F;
      dqY[IP] = ZERO_F;
      dqY[IU] = ZERO_F;
      dqY[IV] = ZERO_F;

      return;
    }

    if (slope_type==1 || slope_type==2) {  // minmod or average

      slope_unsplit_hydro_2d_scalar( q[ID], qPlusX[ID], qMinusX[ID], qPlusY[ID], qMinusY[ID], &(dqX[ID]), &(dqY[ID]));
      slope_unsplit_hydro_2d_scalar( q[IP], qPlusX[IP], qMinusX[IP], qPlusY[IP], qMinusY[IP], &(dqX[IP]), &(dqY[IP]));
      slope_unsplit_hydro_2d_scalar( q[IU], qPlusX[IU], qMinusX[IU], qPlusY[IU], qMinusY[IU], &(dqX[IU]), &(dqY[IU]));
      slope_unsplit_hydro_2d_scalar( q[IV], qPlusX[IV], qMinusX[IV], qPlusY[IV], qMinusY[IV], &(dqX[IV]), &(dqY[IV]));

    } // end slope_type == 1 or 2
  
  } // slope_unsplit_hydro_2d

  /**
   * Compute cell fluxes from the Godunov state
   * \param[in]  qgdnv input Godunov state
   * \param[out] flux  output flux vector
   */
  KOKKOS_INLINE_FUNCTION
  void cmpflx(const MHDState& qgdnv, 
	      MHDState& flux) const
  {
    real_t gamma0 = params.settings.gamma0;

    // Compute fluxes
    // Mass density
    flux[ID] = qgdnv[ID] * qgdnv[IU];
  
    // Normal momentum
    flux[IU] = flux[ID] * qgdnv[IU] + qgdnv[IP];
  
    // Transverse momentum
    flux[IV] = flux[ID] * qgdnv[IV];

    // Total energy
    real_t entho = ONE_F / (gamma0 - ONE_F);
    real_t ekin;
    ekin = 0.5 * qgdnv[ID] * (qgdnv[IU]*qgdnv[IU] + qgdnv[IV]*qgdnv[IV]);
  
    real_t etot = qgdnv[IP] * entho + ekin;
    flux[IP] = qgdnv[IU] * (etot + qgdnv[IP]);

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
    double u = qState[IU];
    double v = qState[IV];
    double w = qState[IW];
    
    d=qState[ID];  p=qState[IP]; 
    a=qState[IBX]; b=qState[IBY]; c=qState[IBZ];
    
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
    
    d=qvar[ID];  p=qvar[IP]; 
    a=qvar[IBX]; b=qvar[IBY]; c=qvar[IBZ];
    
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
  void find_mhd_flux(const MHDState &qvar, 
		     MHDState &cvar,
		     MHDState &ff) const
  {
    
    // ISOTHERMAL
    real_t cIso = params.settings.cIso;
    real_t p;
    if (cIso>0) {
      // recompute pressure
      p = qvar[ID]*cIso*cIso;
    } else {
      p = qvar[IP];
    }
    // end ISOTHERMAL
    
    // local variables
    const real_t entho = ONE_F / (params.settings.gamma0 - ONE_F);
    
    real_t d, u, v, w, a, b, c;
    d=qvar[ID]; 
    u=qvar[IU]; v=qvar[IV]; w=qvar[IW];
    a=qvar[IBX]; b=qvar[IBY]; c=qvar[IBZ];
    
    real_t ecin = 0.5*(u*u+v*v+w*w)*d;
    real_t emag = 0.5*(a*a+b*b+c*c);
    real_t etot = p*entho+ecin+emag;
    real_t ptot = p + emag;
    
    // compute conservative variables
    cvar[ID]  = d;
    cvar[IP]  = etot;
    cvar[IU]  = d*u;
    cvar[IV]  = d*v;
    cvar[IW]  = d*w;
    cvar[IBX] = a;
    cvar[IBY] = b;
    cvar[IBZ] = c;
    
    // compute fluxes
    ff[ID]  = d*u;
    ff[IP]  = (etot+ptot)*u-a*(a*u+b*v+c*w);
    ff[IU]  = d*u*u-a*a+ptot; /* *** WARNING pressure included *** */
    ff[IV]  = d*u*v-a*b;
    ff[IW]  = d*u*w-a*c;
    ff[IBX] = 0.0;
    ff[IBY] = b*u-a*v;
    ff[IBZ] = c*u-a*w;
    
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
    real_t bx_mean = 0.5 * ( qleft[IBX] + qright[IBX] );

    qleft[IBX]  = bx_mean;
    qright[IBX] = bx_mean;
    
    MHDState uleft,  fleft;
    MHDState uright, fright;
    
    find_mhd_flux(qleft ,uleft ,fleft );
    find_mhd_flux(qright,uright,fright);
    
    // find the largest eigenvalue in the normal direction to the interface
    real_t cfleft  = find_speed_fast<IX>(qleft);
    real_t cfright = find_speed_fast<IX>(qright);
    
    real_t vleft =qleft[IU];
    real_t vright=qright[IU];
    real_t sl=fmin ( fmin (vleft,vright) - fmax (cfleft,cfright) , 0.0);
    real_t sr=fmax ( fmax (vleft,vright) + fmax (cfleft,cfright) , 0.0);
    
    // the hll flux
    flux[ID] = (sr*fleft[ID]-sl*fright[ID]+
	      sr*sl*(uright[ID]-uleft[ID]))/(sr-sl);
    flux[IP] = (sr*fleft[IP]-sl*fright[IP]+
	      sr*sl*(uright[IP]-uleft[IP]))/(sr-sl);
    flux[IU] = (sr*fleft[IU]-sl*fright[IU]+
	      sr*sl*(uright[IU]-uleft[IU]))/(sr-sl);
    flux[IV] = (sr*fleft[IV]-sl*fright[IV]+
	      sr*sl*(uright[IV]-uleft[IV]))/(sr-sl);
    flux[IW] = (sr*fleft[IW]-sl*fright[IW]+
	      sr*sl*(uright[IW]-uleft[IW]))/(sr-sl);
    flux[IBX] = (sr*fleft[IBX]-sl*fright[IBX]+
	       sr*sl*(uright[IBX]-uleft[IBX]))/(sr-sl);
    flux[IBY] = (sr*fleft[IBY]-sl*fright[IBY]+
	       sr*sl*(uright[IBY]-uleft[IBY]))/(sr-sl);
    flux[IBZ] = (sr*fleft[IBZ]-sl*fright[IBZ]+
	       sr*sl*(uright[IBZ]-uleft[IBZ]))/(sr-sl);

    
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
   * flux[IU] (as in RAMSES). This will need to be modified in the
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
    real_t a    = 0.5 * ( qleft[IBX] + qright[IBX] );
    real_t sgnm = (a >= 0) ? ONE_F : -ONE_F;
    
    qleft [IBX]  = a; 
    qright[IBX]  = a;
    
    // ISOTHERMAL
    real_t cIso = params.settings.cIso;
    if (cIso > 0) {
      // recompute pressure
      qleft [IP] = qleft [ID]*cIso*cIso;
      qright[IP] = qright[ID]*cIso*cIso;
    } // end ISOTHERMAL
    
    // left variables
    real_t rl, pl, ul, vl, wl, bl, cl;
    rl = qleft[ID]; //rl = fmax(qleft[ID], static_cast<real_t>(gParams.smallr)    );  
    pl = qleft[IP]; //pl = fmax(qleft[IP], static_cast<real_t>(rl*gParams.smallp) ); 
    ul = qleft[IU];  vl = qleft[IV];  wl = qleft[IW]; 
    bl = qleft[IBY];  cl = qleft[IBZ];
    real_t ecinl = 0.5 * (ul*ul + vl*vl + wl*wl) * rl;
    real_t emagl = 0.5 * ( a*a  + bl*bl + cl*cl);
    real_t etotl = pl*entho + ecinl + emagl;
    real_t ptotl = pl + emagl;
    real_t vdotbl= ul*a + vl*bl + wl*cl;
    
    // right variables
    real_t rr, pr, ur, vr, wr, br, cr;
    rr = qright[ID]; //rr = fmax(qright[ID], static_cast<real_t>( gParams.smallr) );
    pr = qright[IP]; //pr = fmax(qright[IP], static_cast<real_t>( rr*gParams.smallp) ); 
    ur = qright[IU];  vr=qright[IV];  wr = qright[IW]; 
    br = qright[IBY];  cr=qright[IBZ];
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
    flux[ID] = ro*uo;
    flux[IP] = (etoto+ptoto)*uo-a*vdotbo;
    flux[IU] = ro*uo*uo-a*a+ptoto; /* *** WARNING *** : ptoto used here (this is only valid for cartesian geometry) ! */
    flux[IV] = ro*uo*vo-a*bo;
    flux[IW] = ro*uo*wo-a*co;
    flux[IBX] = 0.0;
    flux[IBY] = bo*uo-a*vo;
    flux[IBZ] = co*uo-a*wo;
    
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
  real_t compute_emf(MHDState (&qEdge) [4], real_t xPos=0) const
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
    qLL[ID] = qRT[ID];
    qRL[ID] = qLT[ID];
    qLR[ID] = qRB[ID];
    qRR[ID] = qLB[ID];

    // pressure
    // ISOTHERMAL
    real_t cIso = params.settings.cIso;
    if (cIso > 0) {
      qLL[IP] = qLL[ID]*cIso*cIso;
      qRL[IP] = qRL[ID]*cIso*cIso;
      qLR[IP] = qLR[ID]*cIso*cIso;
      qRR[IP] = qRR[ID]*cIso*cIso;
    } else {
      qLL[IP] = qRT[IP];
      qRL[IP] = qLT[IP];
      qLR[IP] = qRB[IP];
      qRR[IP] = qLB[IP];
    }

    // iu, iv : parallel velocity indexes
    // iw     : orthogonal velocity index
    // ia, ib, ic : idem for magnetic field
    //int iu, iv, iw, ia, ib, ic;
    if (emfDir == EMFZ) {

      //iu = IU; iv = IV; iw = IW;
      //ia = IA; ib = IB, ic = IC;
      
      // First parallel velocity 
      qLL[IU] = qRT[IU];
      qRL[IU] = qLT[IU];
      qLR[IU] = qRB[IU];
      qRR[IU] = qLB[IU];
      
      // Second parallel velocity 
      qLL[IV] = qRT[IV];
      qRL[IV] = qLT[IV];
      qLR[IV] = qRB[IV];
      qRR[IV] = qLB[IV];
      
      // First parallel magnetic field (enforce continuity)
      qLL[IBX] = HALF_F * ( qRT[IBX] + qLT[IBX] );
      qRL[IBX] = HALF_F * ( qRT[IBX] + qLT[IBX] );
      qLR[IBX] = HALF_F * ( qRB[IBX] + qLB[IBX] );
      qRR[IBX] = HALF_F * ( qRB[IBX] + qLB[IBX] );
      
      // Second parallel magnetic field (enforce continuity)
      qLL[IBY] = HALF_F * ( qRT[IBY] + qRB[IBY] );
      qRL[IBY] = HALF_F * ( qLT[IBY] + qLB[IBY] );
      qLR[IBY] = HALF_F * ( qRT[IBY] + qRB[IBY] );
      qRR[IBY] = HALF_F * ( qLT[IBY] + qLB[IBY] );
      
      // Orthogonal velocity 
      qLL[IW] = qRT[IW];
      qRL[IW] = qLT[IW];
      qLR[IW] = qRB[IW];
      qRR[IW] = qLB[IW];
      
      // Orthogonal magnetic Field
      qLL[IBZ] = qRT[IBZ];
      qRL[IBZ] = qLT[IBZ];
      qLR[IBZ] = qRB[IBZ];
      qRR[IBZ] = qLB[IBZ];

    } else if (emfDir == EMFY) {

      //iu = IW; iv = IU; iw = IV;
      //ia = IC; ib = IA, ic = IB;
      
      // First parallel velocity 
      qLL[IU] = qRT[IW];
      qRL[IU] = qLT[IW];
      qLR[IU] = qRB[IW];
      qRR[IU] = qLB[IW];
      
      // Second parallel velocity 
      qLL[IV] = qRT[IU];
      qRL[IV] = qLT[IU];
      qLR[IV] = qRB[IU];
      qRR[IV] = qLB[IU];
      
      // First parallel magnetic field (enforce continuity)
      qLL[IBX] = HALF_F * ( qRT[IBZ] + qLT[IBZ] );
      qRL[IBX] = HALF_F * ( qRT[IBZ] + qLT[IBZ] );
      qLR[IBX] = HALF_F * ( qRB[IBZ] + qLB[IBZ] );
      qRR[IBX] = HALF_F * ( qRB[IBZ] + qLB[IBZ] );
      
      // Second parallel magnetic field (enforce continuity)
      qLL[IBY] = HALF_F * ( qRT[IBX] + qRB[IBX] );
      qRL[IBY] = HALF_F * ( qLT[IBX] + qLB[IBX] );
      qLR[IBY] = HALF_F * ( qRT[IBX] + qRB[IBX] );
      qRR[IBY] = HALF_F * ( qLT[IBX] + qLB[IBX] );
      
      // Orthogonal velocity 
      qLL[IW] = qRT[IV];
      qRL[IW] = qLT[IV];
      qLR[IW] = qRB[IV];
      qRR[IW] = qLB[IV];
      
      // Orthogonal magnetic Field
      qLL[IBZ] = qRT[IBY];
      qRL[IBZ] = qLT[IBY];
      qLR[IBZ] = qRB[IBY];
      qRR[IBZ] = qLB[IBY];
      
    } else { // emfDir == EMFX

      //iu = IV; iv = IW; iw = IU;
      //ia = IB; ib = IC, ic = IA;

      // First parallel velocity 
      qLL[IU] = qRT[IV];
      qRL[IU] = qLT[IV];
      qLR[IU] = qRB[IV];
      qRR[IU] = qLB[IV];
      
      // Second parallel velocity 
      qLL[IV] = qRT[IW];
      qRL[IV] = qLT[IW];
      qLR[IV] = qRB[IW];
      qRR[IV] = qLB[IW];
      
      // First parallel magnetic field (enforce continuity)
      qLL[IBX] = HALF_F * ( qRT[IBY] + qLT[IBY] );
      qRL[IBX] = HALF_F * ( qRT[IBY] + qLT[IBY] );
      qLR[IBX] = HALF_F * ( qRB[IBY] + qLB[IBY] );
      qRR[IBX] = HALF_F * ( qRB[IBY] + qLB[IBY] );
      
      // Second parallel magnetic field (enforce continuity)
      qLL[IBY] = HALF_F * ( qRT[IBZ] + qRB[IBZ] );
      qRL[IBY] = HALF_F * ( qLT[IBZ] + qLB[IBZ] );
      qLR[IBY] = HALF_F * ( qRT[IBZ] + qRB[IBZ] );
      qRR[IBY] = HALF_F * ( qLT[IBZ] + qLB[IBZ] );
      
      // Orthogonal velocity 
      qLL[IW] = qRT[IU];
      qRL[IW] = qLT[IU];
      qLR[IW] = qRB[IU];
      qRR[IW] = qLB[IU];
      
      // Orthogonal magnetic Field
      qLL[IBZ] = qRT[IBX];
      qRL[IBZ] = qLT[IBX];
      qLR[IBZ] = qRB[IBX];
      qRR[IBZ] = qLB[IBX];
    }

  
    // Compute final fluxes
  
    // vx*by - vy*bx at the four edge centers
    real_t eLLRR[4];
    real_t &ELL = eLLRR[ILL];
    real_t &ERL = eLLRR[IRL];
    real_t &ELR = eLLRR[ILR];
    real_t &ERR = eLLRR[IRR];

    ELL = qLL[IU]*qLL[IBY] - qLL[IV]*qLL[IBX];
    ERL = qRL[IU]*qRL[IBY] - qRL[IV]*qRL[IBX];
    ELR = qLR[IU]*qLR[IBY] - qLR[IV]*qLR[IBX];
    ERR = qRR[IU]*qRR[IBY] - qRR[IV]*qRR[IBX];

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
    // 	  emf += shear * qLL[IBY];
    // 	} else {
    // 	  emf += shear * qRR[IBY];
    // 	}
    //   }
    //   if (emfDir==EMFZ) {
    // 	real_t shear = -1.5 * params.Omega0 * (xPos - params[ID]x/2);
    // 	if (shear>0) {
    // 	  emf -= shear * qLL[IBX];
    // 	} else {
    // 	  emf -= shear * qRR[IBX];
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
  real_t mag_riemann2d_hlld(const MHDState (&qLLRR)[4],
			    real_t eLLRR[4]) const
  {

    // alias reference to input arrays
    const MHDState &qLL = qLLRR[ILL];
    const MHDState &qRL = qLLRR[IRL];
    const MHDState &qLR = qLLRR[ILR];
    const MHDState &qRR = qLLRR[IRR];

    real_t &ELL = eLLRR[ILL];
    real_t &ERL = eLLRR[IRL];
    real_t &ELR = eLLRR[ILR];
    real_t &ERR = eLLRR[IRR];
    //real_t ELL,ERL,ELR,ERR;

    const real_t &rLL=qLL[ID]; const real_t &pLL=qLL[IP]; 
    const real_t &uLL=qLL[IU]; const real_t &vLL=qLL[IV]; 
    const real_t &aLL=qLL[IBX]; const real_t &bLL=qLL[IBY] ; const real_t &cLL=qLL[IBZ];
  
    const real_t &rLR=qLR[ID]; const real_t &pLR=qLR[IP]; 
    const real_t &uLR=qLR[IU]; const real_t &vLR=qLR[IV]; 
    const real_t &aLR=qLR[IBX]; const real_t &bLR=qLR[IBY] ; const real_t &cLR=qLR[IBZ];
  
    const real_t &rRL=qRL[ID]; const real_t &pRL=qRL[IP]; 
    const real_t &uRL=qRL[IU]; const real_t &vRL=qRL[IV]; 
    const real_t &aRL=qRL[IBX]; const real_t &bRL=qRL[IBY] ; const real_t &cRL=qRL[IBZ];

    const real_t &rRR=qRR[ID]; const real_t &pRR=qRR[IP]; 
    const real_t &uRR=qRR[IU]; const real_t &vRR=qRR[IV]; 
    const real_t &aRR=qRR[IBX]; const real_t &bRR=qRR[IBY] ; const real_t &cRR=qRR[IBZ];
  
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
