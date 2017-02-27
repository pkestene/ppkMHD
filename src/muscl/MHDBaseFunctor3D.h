#ifndef MHD_BASE_FUNCTOR_3D_H_
#define MHD_BASE_FUNCTOR_3D_H_

#include "kokkos_shared.h"

#include "HydroParams.h"
#include "HydroState.h"

namespace ppkMHD { namespace muscl {

/**
 * Base class to derive actual kokkos functor.
 * params is passed by copy.
 */
class MHDBaseFunctor3D
{

public:

  using HydroState = MHDState;
  using DataArray  = DataArray3d;

  MHDBaseFunctor3D(HydroParams params) : params(params) {};
  virtual ~MHDBaseFunctor3D() {};

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
   * Copy data(i,j,k) into q.
   */
  KOKKOS_INLINE_FUNCTION
  void get_state(DataArray data, int i, int j, int k, MHDState& q) const
  {

    q.d  = data(i,j,k, ID);
    q.p  = data(i,j,k, IP);
    q.u  = data(i,j,k, IU);
    q.v  = data(i,j,k, IV);
    q.w  = data(i,j,k, IW);
    q.bx = data(i,j,k, IBX);
    q.by = data(i,j,k, IBY);
    q.bz = data(i,j,k, IBZ);
    
  } // get_state

  /**
   * Copy q into data(i,j,k).
   */
  KOKKOS_INLINE_FUNCTION
  void set_state(DataArray data, int i, int j, int k, const MHDState& q) const
  {

    data(i,j,k, ID)  = q.d;
    data(i,j,k, IP)  = q.p;
    data(i,j,k, IU)  = q.u;
    data(i,j,k, IV)  = q.v;
    data(i,j,k, IW)  = q.w;
    data(i,j,k, IBX) = q.bx;
    data(i,j,k, IBY) = q.by;
    data(i,j,k, IBZ) = q.bz;
    
  } // set_state

  /**
   *
   */
  KOKKOS_INLINE_FUNCTION
  void get_magField(const DataArray& data, int i, int j, int k, BField& b) const
  {

    b.bx = data(i,j,k, IBX);
    b.by = data(i,j,k, IBY);
    b.bz = data(i,j,k, IBZ);
    
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
     * This routine is only used in the 3D UNSPLIT integration and slope_type = 0,1 and 2.
     * 
     * Only slope_type 1 and 2 are supported.
     *
     * \param[in]  q       : current primitive variable
     * \param[in]  qPlusX  : value in the next neighbor cell along XDIR
     * \param[in]  qMinusX : value in the previous neighbor cell along XDIR
     * \param[in]  qPlusY  : value in the next neighbor cell along YDIR
     * \param[in]  qMinusY : value in the previous neighbor cell along YDIR
     * \param[in]  qPlusZ  : value in the next neighbor cell along ZDIR
     * \param[in]  qMinusZ : value in the previous neighbor cell along ZDIR
     * \param[out] dqX     : reference to an array returning the X slopes
     * \param[out] dqY     : reference to an array returning the Y slopes
     * \param[out] dqZ     : reference to an array returning the Z slopes
     *
     */
  KOKKOS_INLINE_FUNCTION
  void slope_unsplit_hydro_3d_scalar(real_t q, 
				     real_t qPlusX,
				     real_t qMinusX,
				     real_t qPlusY,
				     real_t qMinusY,
				     real_t qPlusZ,
				     real_t qMinusZ,
				     real_t *dqX,
				     real_t *dqY,
				     real_t *dqZ) const
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

    // slopes in second coordinate direction
    dlft = slope_type*(q      - qMinusZ);
    drgt = slope_type*(qPlusZ - q      );
    dcen = 0.5 * (qPlusZ - qMinusZ);
    dsgn = (dcen >= ZERO_F) ? ONE_F : -ONE_F;
    slop = fmin( FABS(dlft), FABS(drgt) );
    dlim = slop;
    if ( (dlft*drgt) <= ZERO_F )
      dlim = ZERO_F;
    *dqZ = dsgn * fmin( dlim, FABS(dcen) );

  } // slope_unsplit_hydro_3d_scalar


  /**
   * Compute primitive variables slope (vector dq) from q and its neighbors.
   * This routine is only used in the 3D UNSPLIT integration and slope_type = 0,1,2 and 3.
   * 
   * Note that slope_type is a global variable, located in symbol memory when 
   * using the GPU version.
   *
   * Loosely adapted from RAMSES/hydro/umuscl.f90: subroutine uslope
   * Interface is changed to become cellwise.
   * Only slope_type 1 and 2 are supported.
   *
   * \param[in]  qNb     : array to primitive variable vector state in the neighborhood
   * \param[out] dq      : reference to an array returning the X,Y  and Z slopes
   *
   * 
   */
  KOKKOS_INLINE_FUNCTION
  void slope_unsplit_hydro_3d(const MHDState & q      , 
			      const MHDState & qPlusX , 
			      const MHDState & qMinusX,
			      const MHDState & qPlusY , 
			      const MHDState & qMinusY,
			      const MHDState & qPlusZ , 
			      const MHDState & qMinusZ,
			      MHDState (&dq)[3]) const
  {			
    real_t slope_type = params.settings.slope_type;

    MHDState &dqX = dq[IX];
    MHDState &dqY = dq[IY];
    MHDState &dqZ = dq[IZ];
 
    if (slope_type==0) {

      dqX.d  = ZERO_F; dqY.d  = ZERO_F; dqZ.d  = ZERO_F;
      dqX.p  = ZERO_F; dqY.p  = ZERO_F; dqZ.p  = ZERO_F;
      dqX.u  = ZERO_F; dqY.u  = ZERO_F; dqZ.u  = ZERO_F;
      dqX.v  = ZERO_F; dqY.v  = ZERO_F; dqZ.v  = ZERO_F;
      dqX.w  = ZERO_F; dqY.w  = ZERO_F; dqZ.w  = ZERO_F;
      dqX.bx = ZERO_F; dqY.bx = ZERO_F; dqZ.bx = ZERO_F;
      dqX.by = ZERO_F; dqY.by = ZERO_F; dqZ.by = ZERO_F;
      dqX.bz = ZERO_F; dqY.bz = ZERO_F; dqZ.bz = ZERO_F;

      return;
    }

    if (slope_type==1 or
	slope_type==2) {  // minmod or average

      slope_unsplit_hydro_3d_scalar(q.d,qPlusX.d,qMinusX.d,qPlusY.d,qMinusY.d,qPlusZ.d,qMinusZ.d, &(dqX.d), &(dqY.d), &(dqZ.d));
      slope_unsplit_hydro_3d_scalar(q.p,qPlusX.p,qMinusX.p,qPlusY.p,qMinusY.p,qPlusZ.p,qMinusZ.p, &(dqX.p), &(dqY.p), &(dqZ.p));
      slope_unsplit_hydro_3d_scalar(q.u,qPlusX.u,qMinusX.u,qPlusY.u,qMinusY.u,qPlusZ.u,qMinusZ.u, &(dqX.u), &(dqY.u), &(dqZ.u));
      slope_unsplit_hydro_3d_scalar(q.v,qPlusX.v,qMinusX.v,qPlusY.v,qMinusY.v,qPlusZ.v,qMinusZ.v, &(dqX.v), &(dqY.v), &(dqZ.v));
      slope_unsplit_hydro_3d_scalar(q.w,qPlusX.w,qMinusX.w,qPlusY.w,qMinusY.w,qPlusZ.w,qMinusZ.w, &(dqX.w), &(dqY.w), &(dqZ.w));
      slope_unsplit_hydro_3d_scalar(q.bx,qPlusX.bx,qMinusX.bx,qPlusY.bx,qMinusY.bx,qPlusZ.bx,qMinusZ.bx, &(dqX.bx), &(dqY.bx), &(dqZ.bx));
      slope_unsplit_hydro_3d_scalar(q.by,qPlusX.by,qMinusX.by,qPlusY.by,qMinusY.by,qPlusZ.by,qMinusZ.by, &(dqX.by), &(dqY.by), &(dqZ.by));
      slope_unsplit_hydro_3d_scalar(q.bz,qPlusX.bz,qMinusX.bz,qPlusY.bz,qMinusY.bz,qPlusZ.bz,qMinusY.bz, &(dqX.bz), &(dqY.bz), &(dqZ.bz));
      
    }
  
  } // slope_unsplit_hydro_3d


  /**
   * slope_unsplit_mhd_3d computes only magnetic field slopes in 3D; hydro
   * slopes are always computed in slope_unsplit_hydro_3d.
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
   * and neighboring cells. There are 15 values (5 values for bf_x along
   * y and z, 5 for bf_y along x and z, 5 for bf_z along x and y).
   * 
   * \param[out] dbf : reference to an array returning magnetic field slopes 
   *
   * \note This routine is called inside trace_unsplit_mhd_3d
   */
  KOKKOS_INLINE_FUNCTION
  void slope_unsplit_mhd_3d(real_t bfNeighbors[15],
			    real_t (&dbf)[3][3]) const
  {			
    /* layout for face centered magnetic field */
    real_t &bfx        = bfNeighbors[0];
    real_t &bfx_yplus  = bfNeighbors[1];
    real_t &bfx_yminus = bfNeighbors[2];
    real_t &bfx_zplus  = bfNeighbors[3];
    real_t &bfx_zminus = bfNeighbors[4];

    real_t &bfy        = bfNeighbors[5];
    real_t &bfy_xplus  = bfNeighbors[6];
    real_t &bfy_xminus = bfNeighbors[7];
    real_t &bfy_zplus  = bfNeighbors[8];
    real_t &bfy_zminus = bfNeighbors[9];
  
    real_t &bfz        = bfNeighbors[10];
    real_t &bfz_xplus  = bfNeighbors[11];
    real_t &bfz_xminus = bfNeighbors[12];
    real_t &bfz_yplus  = bfNeighbors[13];
    real_t &bfz_yminus = bfNeighbors[14];
  

    real_t (&dbfX)[3] = dbf[IX];
    real_t (&dbfY)[3] = dbf[IY];
    real_t (&dbfZ)[3] = dbf[IZ];

    // default values for magnetic field slopes
    for (int nVar=0; nVar<3; ++nVar) {
      dbfX[nVar] = ZERO_F;
      dbfY[nVar] = ZERO_F;
      dbfZ[nVar] = ZERO_F;
    }
  
    /*
     * face-centered magnetic field slopes
     */
    // 1D transverse TVD slopes for face-centered magnetic fields
    real_t xslope_type = FMIN(params.settings.slope_type, 2);
    real_t dlft, drgt, dcen, dsgn, slop, dlim;
    {
      // Bx along direction Y     
      dlft = xslope_type * (bfx       - bfx_yminus);
      drgt = xslope_type * (bfx_yplus - bfx       );
      dcen = HALF_F      * (bfx_yplus - bfx_yminus);
      dsgn = (dcen >= ZERO_F) ? ONE_F : -ONE_F;
      slop = FMIN( FABS(dlft), FABS(drgt) );
      dlim = slop;
      if ( (dlft*drgt) <= ZERO_F )
	dlim = ZERO_F;
      dbfY[IX] = dsgn * FMIN( dlim, FABS(dcen) );
      // Bx along direction Z    
      dlft = xslope_type * (bfx       - bfx_zminus);
      drgt = xslope_type * (bfx_zplus - bfx       );
      dcen = HALF_F      * (bfx_zplus - bfx_zminus);
      dsgn = (dcen >= ZERO_F) ? ONE_F : -ONE_F;
      slop = FMIN( FABS(dlft), FABS(drgt) );
      dlim = slop;
      if ( (dlft*drgt) <= ZERO_F )
	dlim = ZERO_F;
      dbfZ[IX] = dsgn * FMIN( dlim, FABS(dcen) );
      
      // By along direction X
      dlft = xslope_type * (bfy       - bfy_xminus);
      drgt = xslope_type * (bfy_xplus - bfy       );
      dcen = HALF_F      * (bfy_xplus - bfy_xminus);
      dsgn = (dcen >= ZERO_F) ? ONE_F : -ONE_F;
      slop = FMIN( FABS(dlft), FABS(drgt) );
      dlim = slop;
      if( (dlft*drgt) <= ZERO_F )
	dlim=ZERO_F;
      dbfX[IY] = dsgn * FMIN( dlim, FABS(dcen) );
      // By along direction Z
      dlft = xslope_type * (bfy       - bfy_zminus);
      drgt = xslope_type * (bfy_zplus - bfy       );
      dcen = HALF_F      * (bfy_zplus - bfy_zminus);
      dsgn = (dcen >= ZERO_F) ? ONE_F : -ONE_F;
      slop = FMIN( FABS(dlft), FABS(drgt) );
      dlim = slop;
      if( (dlft*drgt) <= ZERO_F )
	dlim=ZERO_F;
      dbfZ[IY] = dsgn * FMIN( dlim, FABS(dcen) );

      // Bz along direction X
      dlft = xslope_type * (bfz       - bfz_xminus);
      drgt = xslope_type * (bfz_xplus - bfz       );
      dcen = HALF_F      * (bfz_xplus - bfz_xminus);
      dsgn = (dcen >= ZERO_F) ? ONE_F : -ONE_F;
      slop = FMIN( FABS(dlft), FABS(drgt) );
      dlim = slop;
      if( (dlft*drgt) <= ZERO_F )
	dlim=ZERO_F;
      dbfX[IZ] = dsgn * FMIN( dlim, FABS(dcen) );
      // Bz along direction Y
      dlft = xslope_type * (bfz       - bfz_yminus);
      drgt = xslope_type * (bfz_yplus - bfz       );
      dcen = HALF_F      * (bfz_yplus - bfz_yminus);
      dsgn = (dcen >= ZERO_F) ? ONE_F : -ONE_F;
      slop = FMIN( FABS(dlft), FABS(drgt) );
      dlim = slop;
      if( (dlft*drgt) <= ZERO_F )
	dlim=ZERO_F;
      dbfY[IZ] = dsgn * FMIN( dlim, FABS(dcen) );

    }

  } // slope_unsplit_mhd_3d


  /**
   * This another implementation of trace computations simpler than 
   * trace_unsplit_mhd_3d.
   *
   * By simpler, we mean to design a device function that could lead to better
   * ressource utilization and thus better performances (hopefully).
   *
   * To achieve this goal, several modifications are brought (compared to 
   * trace_unsplit_mhd_3d) :
   * - hydro slopes (call to slope_unsplit_hydro_3d is done outside)
   * - face-centered magnetic field slopes is done outside and before, so it is
   *   an input now
   * - electric field computation is done outside and before (probably in a 
   *   separate CUDA kernel as for the GPU version), so it is now an input 
   *
   *
   */
  KOKKOS_INLINE_FUNCTION
  void trace_unsplit_mhd_3d_simpler(MHDState q,
				    MHDState dq[THREE_D],
				    real_t bfNb[THREE_D*2], /* 2 faces per direction*/
				    real_t dbf[12],
				    real_t elecFields[THREE_D][2][2],
				    real_t dtdx,
				    real_t dtdy,
				    real_t dtdz,
				    real_t xPos,
				    MHDState (&qm)[THREE_D], 
				    MHDState (&qp)[THREE_D],
				    MHDState (&qEdge)[4][3]) const
  {
  
    // inputs
    // alias to electric field components
    real_t (&Ex)[2][2] = elecFields[IX];
    real_t (&Ey)[2][2] = elecFields[IY];
    real_t (&Ez)[2][2] = elecFields[IZ];

    // outputs
    // alias for q on cell edge (as defined in DUMSES trace3d routine)
    MHDState &qRT_X = qEdge[0][0];
    MHDState &qRB_X = qEdge[1][0];
    MHDState &qLT_X = qEdge[2][0];
    MHDState &qLB_X = qEdge[3][0];

    MHDState &qRT_Y = qEdge[0][1];
    MHDState &qRB_Y = qEdge[1][1];
    MHDState &qLT_Y = qEdge[2][1];
    MHDState &qLB_Y = qEdge[3][1];

    MHDState &qRT_Z = qEdge[0][2];
    MHDState &qRB_Z = qEdge[1][2];
    MHDState &qLT_Z = qEdge[2][2];
    MHDState &qLB_Z = qEdge[3][2];

    real_t gamma  = params.settings.gamma0;
    real_t smallR = params.settings.smallr;
    real_t smallp = params.settings.smallp;
    real_t Omega0 = params.settings.Omega0;
    real_t dx     = params.dx;

    // Edge centered electric field in X, Y and Z directions
    real_t &ELL = Ex[0][0];
    real_t &ELR = Ex[0][1];
    real_t &ERL = Ex[1][0];
    real_t &ERR = Ex[1][1];

    real_t &FLL = Ey[0][0];
    real_t &FLR = Ey[0][1];
    real_t &FRL = Ey[1][0];
    real_t &FRR = Ey[1][1];
  
    real_t &GLL = Ez[0][0];
    real_t &GLR = Ez[0][1];
    real_t &GRL = Ez[1][0];
    real_t &GRR = Ez[1][1];
  
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
    real_t AL =  bfNb[0];
    real_t AR =  bfNb[1];
    real_t BL =  bfNb[2];
    real_t BR =  bfNb[3];
    real_t CL =  bfNb[4];
    real_t CR =  bfNb[5];

    // Cell centered TVD slopes in X direction
    real_t& drx = dq[IX].d;  drx *= HALF_F;
    real_t& dpx = dq[IX].p;  dpx *= HALF_F;
    real_t& dux = dq[IX].u;  dux *= HALF_F;
    real_t& dvx = dq[IX].v;  dvx *= HALF_F;
    real_t& dwx = dq[IX].w;  dwx *= HALF_F;
    real_t& dCx = dq[IX].bz;  dCx *= HALF_F;
    real_t& dBx = dq[IX].by;  dBx *= HALF_F;
  
    // Cell centered TVD slopes in Y direction
    real_t& dry = dq[IY].d;  dry *= HALF_F;
    real_t& dpy = dq[IY].p;  dpy *= HALF_F;
    real_t& duy = dq[IY].u;  duy *= HALF_F;
    real_t& dvy = dq[IY].v;  dvy *= HALF_F;
    real_t& dwy = dq[IY].w;  dwy *= HALF_F;
    real_t& dCy = dq[IY].bz;  dCy *= HALF_F;
    real_t& dAy = dq[IY].bx;  dAy *= HALF_F;

    // Cell centered TVD slopes in Z direction
    real_t& drz = dq[IZ].d;  drz *= HALF_F;
    real_t& dpz = dq[IZ].p;  dpz *= HALF_F;
    real_t& duz = dq[IZ].u;  duz *= HALF_F;
    real_t& dvz = dq[IZ].v;  dvz *= HALF_F;
    real_t& dwz = dq[IZ].w;  dwz *= HALF_F;
    real_t& dAz = dq[IZ].bx;  dAz *= HALF_F;
    real_t& dBz = dq[IZ].by;  dBz *= HALF_F;


    // Face centered TVD slopes in transverse direction
    real_t dALy = HALF_F * dbf[0];
    real_t dALz = HALF_F * dbf[1];
    real_t dBLx = HALF_F * dbf[2];
    real_t dBLz = HALF_F * dbf[3];
    real_t dCLx = HALF_F * dbf[4];
    real_t dCLy = HALF_F * dbf[5];

    real_t dARy = HALF_F * dbf[6];
    real_t dARz = HALF_F * dbf[7];
    real_t dBRx = HALF_F * dbf[8];
    real_t dBRz = HALF_F * dbf[9];
    real_t dCRx = HALF_F * dbf[10];
    real_t dCRy = HALF_F * dbf[11];

    // Cell centered slopes in normal direction
    real_t dAx = HALF_F * (AR - AL);
    real_t dBy = HALF_F * (BR - BL);
    real_t dCz = HALF_F * (CR - CL);

    // Source terms (including transverse derivatives)
    real_t sr0, su0, sv0, sw0, sp0, sA0, sB0, sC0;
    real_t sAL0, sAR0, sBL0, sBR0, sCL0, sCR0;

    if (true /*cartesian*/) {

      sr0 = (-u*drx-dux*r)              *dtdx + (-v*dry-dvy*r)              *dtdy + (-w*drz-dwz*r)              *dtdz;
      su0 = (-u*dux-(dpx+B*dBx+C*dCx)/r)*dtdx + (-v*duy+B*dAy/r)            *dtdy + (-w*duz+C*dAz/r)            *dtdz; 
      sv0 = (-u*dvx+A*dBx/r)            *dtdx + (-v*dvy-(dpy+A*dAy+C*dCy)/r)*dtdy + (-w*dvz+C*dBz/r)            *dtdz;
      sw0 = (-u*dwx+A*dCx/r)            *dtdx + (-v*dwy+B*dCy/r)            *dtdy + (-w*dwz-(dpz+A*dAz+B*dBz)/r)*dtdz; 
      sp0 = (-u*dpx-dux*gamma*p)        *dtdx + (-v*dpy-dvy*gamma*p)        *dtdy + (-w*dpz-dwz*gamma*p)        *dtdz;
      sA0 =                                     (u*dBy+B*duy-v*dAy-A*dvy)   *dtdy + (u*dCz+C*duz-w*dAz-A*dwz)   *dtdz;
      sB0 = (v*dAx+A*dvx-u*dBx-B*dux)   *dtdx +                                     (v*dCz+C*dvz-w*dBz-B*dwz)   *dtdz; 
      sC0 = (w*dAx+A*dwx-u*dCx-C*dux)   *dtdx + (w*dBy+B*dwy-v*dCy-C*dvy)   *dtdy;
      if (Omega0>0) {
	real_t shear = -1.5 * Omega0 *xPos;
	sr0 = sr0 -  shear*dry*dtdy;
	su0 = su0 -  shear*duy*dtdy;
	sv0 = sv0 -  shear*dvy*dtdy;
	sw0 = sw0 -  shear*dwy*dtdy;
	sp0 = sp0 -  shear*dpy*dtdy;
	sA0 = sA0 -  shear*dAy*dtdy;
	sB0 = sB0 + (shear*dAx - 1.5 * Omega0 * A *dx)*dtdx + shear*dBz*dtdz;
	sC0 = sC0 -  shear*dCy*dtdy;
      }
	
      // Face-centered B-field
      sAL0 = +(GLR-GLL)*dtdy*HALF_F -(FLR-FLL)*dtdz*HALF_F;
      sAR0 = +(GRR-GRL)*dtdy*HALF_F -(FRR-FRL)*dtdz*HALF_F;
      sBL0 = -(GRL-GLL)*dtdx*HALF_F +(ELR-ELL)*dtdz*HALF_F;
      sBR0 = -(GRR-GLR)*dtdx*HALF_F +(ERR-ERL)*dtdz*HALF_F;
      sCL0 = +(FRL-FLL)*dtdx*HALF_F -(ERL-ELL)*dtdy*HALF_F;
      sCR0 = +(FRR-FLR)*dtdx*HALF_F -(ERR-ELR)*dtdy*HALF_F;

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
    CL = CL + sCL0;
    CR = CR + sCR0;

    // Face averaged right state at left interface
    qp[0].d = r - drx;
    qp[0].u = u - dux;
    qp[0].v = v - dvx;
    qp[0].w = w - dwx;
    qp[0].p = p - dpx;
    qp[0].bx = AL;
    qp[0].by = B - dBx;
    qp[0].bz = C - dCx;
    qp[0].d = FMAX(smallR,  qp[0].d);
    qp[0].p = FMAX(smallp /** qp[0].d*/, qp[0].p);
  
    // Face averaged left state at right interface
    qm[0].d = r + drx;
    qm[0].u = u + dux;
    qm[0].v = v + dvx;
    qm[0].w = w + dwx;
    qm[0].p = p + dpx;
    qm[0].bx = AR;
    qm[0].by = B + dBx;
    qm[0].bz = C + dCx;
    qm[0].d = FMAX(smallR,  qm[0].d);
    qm[0].p = FMAX(smallp /** qm[0].d*/, qm[0].p);

    // Face averaged top state at bottom interface
    qp[1].d = r - dry;
    qp[1].u = u - duy;
    qp[1].v = v - dvy;
    qp[1].w = w - dwy;
    qp[1].p = p - dpy;
    qp[1].bx = A - dAy;
    qp[1].by = BL;
    qp[1].bz = C - dCy;
    qp[1].d = FMAX(smallR,  qp[1].d);
    qp[1].p = FMAX(smallp /** qp[1].d*/, qp[1].p);
  
    // Face averaged bottom state at top interface
    qm[1].d = r + dry;
    qm[1].u = u + duy;
    qm[1].v = v + dvy;
    qm[1].w = w + dwy;
    qm[1].p = p + dpy;
    qm[1].bx = A + dAy;
    qm[1].by = BR;
    qm[1].bz = C + dCy;
    qm[1].d = FMAX(smallR,  qm[1].d);
    qm[1].p = FMAX(smallp /** qm[1].d*/, qm[1].p);
  
    // Face averaged front state at back interface
    qp[2].d = r - drz;
    qp[2].u = u - duz;
    qp[2].v = v - dvz;
    qp[2].w = w - dwz;
    qp[2].p = p - dpz;
    qp[2].bx = A - dAz;
    qp[2].by = B - dBz;
    qp[2].bz = CL;
    qp[2].d = FMAX(smallR,  qp[2].d);
    qp[2].p = FMAX(smallp /** qp[2].d*/, qp[2].p);
  
    // Face averaged back state at front interface
    qm[2].d = r + drz;
    qm[2].u = u + duz;
    qm[2].v = v + dvz;
    qm[2].w = w + dwz;
    qm[2].p = p + dpz;
    qm[2].bx = A + dAz;
    qm[2].by = B + dBz;
    qm[2].bz = CR;
    qm[2].d = FMAX(smallR,  qm[2].d);
    qm[2].p = FMAX(smallp /** qm[2].d*/, qm[2].p);

    // X-edge averaged right-top corner state (RT->LL)
    qRT_X.d = r + (+dry+drz);
    qRT_X.u = u + (+duy+duz);
    qRT_X.v = v + (+dvy+dvz);
    qRT_X.w = w + (+dwy+dwz);
    qRT_X.p = p + (+dpy+dpz);
    qRT_X.bx = A + (+dAy+dAz);
    qRT_X.by = BR+ (   +dBRz);
    qRT_X.bz = CR+ (+dCRy   );
    qRT_X.d = FMAX(smallR,  qRT_X.d);
    qRT_X.p = FMAX(smallp /** qRT_X.d*/, qRT_X.p);
  
    // X-edge averaged right-bottom corner state (RB->LR)
    qRB_X.d = r + (+dry-drz);
    qRB_X.u = u + (+duy-duz);
    qRB_X.v = v + (+dvy-dvz);
    qRB_X.w = w + (+dwy-dwz);
    qRB_X.p = p + (+dpy-dpz);
    qRB_X.bx = A + (+dAy-dAz);
    qRB_X.by = BR+ (   -dBRz);
    qRB_X.bz = CL+ (+dCLy   );
    qRB_X.d = FMAX(smallR,  qRB_X.d);
    qRB_X.p = FMAX(smallp /** qRB_X.d*/, qRB_X.p);
  
    // X-edge averaged left-top corner state (LT->RL)
    qLT_X.d = r + (-dry+drz);
    qLT_X.u = u + (-duy+duz);
    qLT_X.v = v + (-dvy+dvz);
    qLT_X.w = w + (-dwy+dwz);
    qLT_X.p = p + (-dpy+dpz);
    qLT_X.bx = A + (-dAy+dAz);
    qLT_X.by = BL+ (   +dBLz);
    qLT_X.bz = CR+ (-dCRy   );
    qLT_X.d = FMAX(smallR,  qLT_X.d);
    qLT_X.p = FMAX(smallp /** qLT_X.d*/, qLT_X.p);
  
    // X-edge averaged left-bottom corner state (LB->RR)
    qLB_X.d = r + (-dry-drz);
    qLB_X.u = u + (-duy-duz);
    qLB_X.v = v + (-dvy-dvz);
    qLB_X.w = w + (-dwy-dwz);
    qLB_X.p = p + (-dpy-dpz);
    qLB_X.bx = A + (-dAy-dAz);
    qLB_X.by = BL+ (   -dBLz);
    qLB_X.bz = CL+ (-dCLy   );
    qLB_X.d = FMAX(smallR,  qLB_X.d);
    qLB_X.p = FMAX(smallp /** qLB_X.d*/, qLB_X.p);
  
    // Y-edge averaged right-top corner state (RT->LL)
    qRT_Y.d = r + (+drx+drz);
    qRT_Y.u = u + (+dux+duz);
    qRT_Y.v = v + (+dvx+dvz);
    qRT_Y.w = w + (+dwx+dwz);
    qRT_Y.p = p + (+dpx+dpz);
    qRT_Y.bx = AR+ (   +dARz);
    qRT_Y.by = B + (+dBx+dBz);
    qRT_Y.bz = CR+ (+dCRx   );
    qRT_Y.d = FMAX(smallR,  qRT_Y.d);
    qRT_Y.p = FMAX(smallp /** qRT_Y.d*/, qRT_Y.p);
  
    // Y-edge averaged right-bottom corner state (RB->LR)
    qRB_Y.d = r + (+drx-drz);
    qRB_Y.u = u + (+dux-duz);
    qRB_Y.v = v + (+dvx-dvz);
    qRB_Y.w = w + (+dwx-dwz);
    qRB_Y.p = p + (+dpx-dpz);
    qRB_Y.bx = AR+ (   -dARz);
    qRB_Y.by = B + (+dBx-dBz);
    qRB_Y.bz = CL+ (+dCLx   );
    qRB_Y.d = FMAX(smallR,  qRB_Y.d);
    qRB_Y.p = FMAX(smallp /** qRB_Y.d*/, qRB_Y.p);
  
    // Y-edge averaged left-top corner state (LT->RL)
    qLT_Y.d = r + (-drx+drz);
    qLT_Y.u = u + (-dux+duz);
    qLT_Y.v = v + (-dvx+dvz);
    qLT_Y.w = w + (-dwx+dwz);
    qLT_Y.p = p + (-dpx+dpz);
    qLT_Y.bx = AL+ (   +dALz);
    qLT_Y.by = B + (-dBx+dBz);
    qLT_Y.bz = CR+ (-dCRx   );
    qLT_Y.d = FMAX(smallR,  qLT_Y.d);
    qLT_Y.p = FMAX(smallp /** qLT_Y.d*/, qLT_Y.p);
  
    // Y-edge averaged left-bottom corner state (LB->RR)
    qLB_Y.d = r + (-drx-drz);
    qLB_Y.u = u + (-dux-duz);
    qLB_Y.v = v + (-dvx-dvz);
    qLB_Y.w = w + (-dwx-dwz);
    qLB_Y.p = p + (-dpx-dpz);
    qLB_Y.bx = AL+ (   -dALz);
    qLB_Y.by = B + (-dBx-dBz);
    qLB_Y.bz = CL+ (-dCLx   );
    qLB_Y.d = FMAX(smallR,  qLB_Y.d);
    qLB_Y.p = FMAX(smallp /** qLB_Y.d*/, qLB_Y.p);
  
    // Z-edge averaged right-top corner state (RT->LL)
    qRT_Z.d = r + (+drx+dry);
    qRT_Z.u = u + (+dux+duy);
    qRT_Z.v = v + (+dvx+dvy);
    qRT_Z.w = w + (+dwx+dwy);
    qRT_Z.p = p + (+dpx+dpy);
    qRT_Z.bx = AR+ (   +dARy);
    qRT_Z.by = BR+ (+dBRx   );
    qRT_Z.bz = C + (+dCx+dCy);
    qRT_Z.d = FMAX(smallR,  qRT_Z.d);
    qRT_Z.p = FMAX(smallp /** qRT_Z.d*/, qRT_Z.p);
  
    // Z-edge averaged right-bottom corner state (RB->LR)
    qRB_Z.d = r + (+drx-dry);
    qRB_Z.u = u + (+dux-duy);
    qRB_Z.v = v + (+dvx-dvy);
    qRB_Z.w = w + (+dwx-dwy);
    qRB_Z.p = p + (+dpx-dpy);
    qRB_Z.bx = AR+ (   -dARy);
    qRB_Z.by = BL+ (+dBLx   );
    qRB_Z.bz = C + (+dCx-dCy);
    qRB_Z.d = FMAX(smallR,  qRB_Z.d);
    qRB_Z.p = FMAX(smallp /** qRB_Z.d*/, qRB_Z.p);
  
    // Z-edge averaged left-top corner state (LT->RL)
    qLT_Z.d = r + (-drx+dry);
    qLT_Z.u = u + (-dux+duy);
    qLT_Z.v = v + (-dvx+dvy);
    qLT_Z.w = w + (-dwx+dwy);
    qLT_Z.p = p + (-dpx+dpy);
    qLT_Z.bx = AL+ (   +dALy);
    qLT_Z.by = BR+ (-dBRx   );
    qLT_Z.bz = C + (-dCx+dCy);
    qLT_Z.d = FMAX(smallR,  qLT_Z.d);
    qLT_Z.p = FMAX(smallp /** qLT_Z.d*/, qLT_Z.p);
  
    // Z-edge averaged left-bottom corner state (LB->RR)
    qLB_Z.d = r + (-drx-dry);
    qLB_Z.u = u + (-dux-duy);
    qLB_Z.v = v + (-dvx-dvy);
    qLB_Z.w = w + (-dwx-dwy);
    qLB_Z.p = p + (-dpx-dpy);
    qLB_Z.bx = AL+ (   -dALy);
    qLB_Z.by = BL+ (-dBLx   );
    qLB_Z.bz = C + (-dCx-dCy);
    qLB_Z.d = FMAX(smallR,  qLB_Z.d);
    qLB_Z.p = FMAX(smallp /** qLB_Z.d*/, qLB_Z.p);

  } // trace_unsplit_mhd_3d_simpler

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

}; // class MHDBaseFunctor3D

} // namespace muscl

} // namespace ppkMHD

#endif // MHD_BASE_FUNCTOR_3D_H_
