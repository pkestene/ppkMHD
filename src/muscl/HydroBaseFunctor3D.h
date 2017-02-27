#ifndef HYDRO_BASE_FUNCTOR_3D_H_
#define HYDRO_BASE_FUNCTOR_3D_H_

#include "kokkos_shared.h"

#include "HydroParams.h"
#include "HydroState.h"

namespace ppkMHD { namespace muscl {

/**
 * Base class to derive actual kokkos functor for hydro 3D.
 * params is passed by copy.
 */
class HydroBaseFunctor3D
{

public:

  using HydroState = HydroState3d;
  using DataArray  = DataArray3d;
  
HydroBaseFunctor3D(HydroParams params) : params(params) {};
  virtual ~HydroBaseFunctor3D() {};

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
    
    *p = FMAX((gamma0 - ONE_F) * rho * eint, rho * smallp);
    *c = SQRT(gamma0 * (*p) / rho);
    
  } // eos
  
  /**
   * Convert conservative variables (rho, rho*u, rho*v, e) to 
   * primitive variables (rho,u,v,p)
   * @param[in]  u  conservative variables array
   * @param[out] q  primitive    variables array (allocated in calling routine, size is constant nbvar)
   * @param[out] c  local speed of sound
   */
  KOKKOS_INLINE_FUNCTION
  void computePrimitives(const HydroState *u,
			 real_t* c,
			 HydroState *q) const
  {
    real_t gamma0 = params.settings.gamma0;
    real_t smallr = params.settings.smallr;
    real_t smallp = params.settings.smallp;
    
    real_t d, p, ux, uy, uz;
    
    d = fmax(u->d, smallr);
    ux = u->u / d;
    uy = u->v / d;
    uz = u->w / d;
    
    real_t eken = HALF_F * (ux*ux + uy*uy + uz*uz);
    real_t e = u->p / d - eken;
    
    // compute pressure and speed of sound
    p = fmax((gamma0 - 1.0) * d * e, d * smallp);
    *c = sqrt(gamma0 * (p) / d);
    
    q->d = d;
    q->p = p;
    q->u = ux;
    q->v = uy;
    q->w = uz;
    
  } // computePrimitive

  /**
   * Trace computations for unsplit Godunov scheme.
   *
   * \param[in] q          : Primitive variables state.
   * \param[in] qNeighbors : state in the neighbor cells (2 neighbors
   * per dimension, in the following order x+, x-, y+, y-, z+, z-)
   * \param[in] c          : local sound speed.
   * \param[in] dtdx       : dt over dx
   * \param[out] qm        : qm state (one per dimension)
   * \param[out] qp        : qp state (one per dimension)
   */
  KOKKOS_INLINE_FUNCTION
  void trace_unsplit_3d(const HydroState *q, 
			const HydroState *qNeighbors_0,
			const HydroState *qNeighbors_1,
			const HydroState *qNeighbors_2,
			const HydroState *qNeighbors_3,
			const HydroState *qNeighbors_4,
			const HydroState *qNeighbors_5,
			real_t c, 
			real_t dtdx, 
			real_t dtdy,
			real_t dtdz,
			HydroState *qm_x,
			HydroState *qm_y,
			HydroState *qm_z,
			HydroState *qp_x,
			HydroState *qp_y,
			HydroState *qp_z) const
  {
    
    real_t gamma0 = params.settings.gamma0;
    real_t smallr = params.settings.smallr;
    
    // first compute slopes
    HydroState dqX, dqY, dqZ;
    dqX.d = 0.0;
    dqX.p = 0.0;
    dqX.u = 0.0;
    dqX.v = 0.0;
    dqX.w = 0.0;
    
    dqY.d = 0.0;
    dqY.p = 0.0;
    dqY.u = 0.0;
    dqY.v = 0.0;
    dqY.w = 0.0;

    dqZ.d = 0.0;
    dqZ.p = 0.0;
    dqZ.u = 0.0;
    dqZ.v = 0.0;
    dqZ.w = 0.0;

    slope_unsplit_hydro_3d(q, 
			   qNeighbors_0, qNeighbors_1, 
			   qNeighbors_2, qNeighbors_3,
			   qNeighbors_4, qNeighbors_5,
			   &dqX, &dqY, &dqZ);
      
    // Cell centered values
    real_t r =  q->d;
    real_t p =  q->p;
    real_t u =  q->u;
    real_t v =  q->v;
    real_t w =  q->w;
      
    // TVD slopes in all directions
    real_t drx = dqX.d;
    real_t dpx = dqX.p;
    real_t dux = dqX.u;
    real_t dvx = dqX.v;
    real_t dwx = dqX.w;
      
    real_t dry = dqY.d;
    real_t dpy = dqY.p;
    real_t duy = dqY.u;
    real_t dvy = dqY.v;
    real_t dwy = dqY.w;

    real_t drz = dqZ.d;
    real_t dpz = dqZ.p;
    real_t duz = dqZ.u;
    real_t dvz = dqZ.v;
    real_t dwz = dqZ.w;
      
    // source terms (with transverse derivatives)
    real_t sr0 = (-u*drx-dux*r)*dtdx + (-v*dry-dvy*r)*dtdy + (-w*drz-dwz*r)*dtdz;
    real_t su0 = (-u*dux-dpx/r)*dtdx + (-v*duy      )*dtdy + (-w*duz      )*dtdz; 
    real_t sv0 = (-u*dvx      )*dtdx + (-v*dvy-dpy/r)*dtdy + (-w*dvz      )*dtdz;
    real_t sw0 = (-u*dwx      )*dtdx + (-v*dwy      )*dtdy + (-w*dwz-dpz/r)*dtdz; 
    real_t sp0 = (-u*dpx-dux*gamma0*p)*dtdx + (-v*dpy-dvy*gamma0*p)*dtdy + (-w*dpz-dwz*gamma0*p)*dtdz;
       
    // Right state at left interface
    qp_x->d = r - HALF_F*drx + sr0*HALF_F;
    qp_x->p = p - HALF_F*dpx + sp0*HALF_F;
    qp_x->u = u - HALF_F*dux + su0*HALF_F;
    qp_x->v = v - HALF_F*dvx + sv0*HALF_F;
    qp_x->w = w - HALF_F*dwx + sw0*HALF_F;
    qp_x->d = fmax(smallr, qp_x->d);
      
    // Left state at right interface
    qm_x->d = r + HALF_F*drx + sr0*HALF_F;
    qm_x->p = p + HALF_F*dpx + sp0*HALF_F;
    qm_x->u = u + HALF_F*dux + su0*HALF_F;
    qm_x->v = v + HALF_F*dvx + sv0*HALF_F;
    qm_x->w = w + HALF_F*dwx + sw0*HALF_F;
    qm_x->d = fmax(smallr, qm_x->d);
      
    // Top state at bottom interface
    qp_y->d = r - HALF_F*dry + sr0*HALF_F;
    qp_y->p = p - HALF_F*dpy + sp0*HALF_F;
    qp_y->u = u - HALF_F*duy + su0*HALF_F;
    qp_y->v = v - HALF_F*dvy + sv0*HALF_F;
    qp_y->w = w - HALF_F*dwy + sw0*HALF_F;
    qp_y->d = fmax(smallr, qp_y->d);
      
    // Bottom state at top interface
    qm_y->d = r + HALF_F*dry + sr0*HALF_F;
    qm_y->p = p + HALF_F*dpy + sp0*HALF_F;
    qm_y->u = u + HALF_F*duy + su0*HALF_F;
    qm_y->v = v + HALF_F*dvy + sv0*HALF_F;
    qm_y->w = w + HALF_F*dwy + sw0*HALF_F;
    qm_y->d = fmax(smallr, qm_y->d);

    // Back state at bottom interface
    qp_z->d = r - HALF_F*drz + sr0*HALF_F;
    qp_z->p = p - HALF_F*dpz + sp0*HALF_F;
    qp_z->u = u - HALF_F*duz + su0*HALF_F;
    qp_z->v = v - HALF_F*dvz + sv0*HALF_F;
    qp_z->w = w - HALF_F*dwz + sw0*HALF_F;
    qp_z->d = fmax(smallr, qp_z->d);
      
    // Front state at top interface
    qm_z->d = r + HALF_F*drz + sr0*HALF_F;
    qm_z->p = p + HALF_F*dpz + sp0*HALF_F;
    qm_z->u = u + HALF_F*duz + su0*HALF_F;
    qm_z->v = v + HALF_F*dvz + sv0*HALF_F;
    qm_z->w = w + HALF_F*dwz + sw0*HALF_F;
    qm_z->d = fmax(smallr, qm_z->d);

  } // trace_unsplit_3d



  /**
   * Trace computations for unsplit Godunov scheme (3d).
   *
   * \param[in] q          : Primitive variables state.
   * \param[in] dqX        : slope along X
   * \param[in] dqY        : slope along Y
   * \param[in] dqZ        : slope along Z
   * \param[in] c          : local sound speed.
   * \param[in] dtdx       : dt over dx
   * \param[in] dtdy       : dt over dy
   * \param[in] dtdz       : dt over dz
   * \param[in] faceId     : which face will be reconstructed
   * \param[out] qface     : q reconstructed state at cell interface
   */
  KOKKOS_INLINE_FUNCTION
  void trace_unsplit_3d_along_dir(const HydroState *q, 
				  const HydroState *dqX,
				  const HydroState *dqY,
				  const HydroState *dqZ,
				  real_t dtdx, 
				  real_t dtdy, 
				  real_t dtdz, 
				  int    faceId,
				  HydroState *qface) const
  {
  
    real_t gamma0 = params.settings.gamma0;
    real_t smallr = params.settings.smallr;

    // Cell centered values
    real_t r =  q->d;
    real_t p =  q->p;
    real_t u =  q->u;
    real_t v =  q->v;
    real_t w =  q->w;
  
    // TVD slopes in all directions
    real_t drx = dqX->d;
    real_t dpx = dqX->p;
    real_t dux = dqX->u;
    real_t dvx = dqX->v;
    real_t dwx = dqX->w;
  
    real_t dry = dqY->d;
    real_t dpy = dqY->p;
    real_t duy = dqY->u;
    real_t dvy = dqY->v;
    real_t dwy = dqY->w;
  
    real_t drz = dqZ->d;
    real_t dpz = dqZ->p;
    real_t duz = dqZ->u;
    real_t dvz = dqZ->v;
    real_t dwz = dqZ->w;
  
    // source terms (with transverse derivatives)
    real_t sr0 = -u*drx-v*dry-w*drz - (dux+dvy+dwz)*r;
    real_t sp0 = -u*dpx-v*dpy-w*dpz - (dux+dvy+dwz)*gamma0*p;
    real_t su0 = -u*dux-v*duy-w*duz - (dpx        )/r;
    real_t sv0 = -u*dvx-v*dvy-w*dvz - (dpy        )/r;
    real_t sw0 = -u*dwx-v*dwy-w*dwz - (dpz        )/r;

    if (faceId == FACE_XMIN) {
      // Right state at left interface
      qface->d = r - HALF_F*drx + sr0*dtdx*HALF_F;
      qface->p = p - HALF_F*dpx + sp0*dtdx*HALF_F;
      qface->u = u - HALF_F*dux + su0*dtdx*HALF_F;
      qface->v = v - HALF_F*dvx + sv0*dtdx*HALF_F;
      qface->w = w - HALF_F*dwx + sw0*dtdx*HALF_F;
      qface->d = fmax(smallr, qface->d);
    }

    if (faceId == FACE_XMAX) {
      // Left state at right interface
      qface->d = r + HALF_F*drx + sr0*dtdx*HALF_F;
      qface->p = p + HALF_F*dpx + sp0*dtdx*HALF_F;
      qface->u = u + HALF_F*dux + su0*dtdx*HALF_F;
      qface->v = v + HALF_F*dvx + sv0*dtdx*HALF_F;
      qface->w = w + HALF_F*dwx + sw0*dtdx*HALF_F;
      qface->d = fmax(smallr, qface->d);
    }
  
    if (faceId == FACE_YMIN) {
      // Top state at bottom interface
      qface->d = r - HALF_F*dry + sr0*dtdy*HALF_F;
      qface->p = p - HALF_F*dpy + sp0*dtdy*HALF_F;
      qface->u = u - HALF_F*duy + su0*dtdy*HALF_F;
      qface->v = v - HALF_F*dvy + sv0*dtdy*HALF_F;
      qface->w = w - HALF_F*dwy + sw0*dtdy*HALF_F;
      qface->d = fmax(smallr, qface->d);
    }

    if (faceId == FACE_YMAX) {
      // Bottom state at top interface
      qface->d = r + HALF_F*dry + sr0*dtdy*HALF_F;
      qface->p = p + HALF_F*dpy + sp0*dtdy*HALF_F;
      qface->u = u + HALF_F*duy + su0*dtdy*HALF_F;
      qface->v = v + HALF_F*dvy + sv0*dtdy*HALF_F;
      qface->w = w + HALF_F*dwy + sw0*dtdy*HALF_F;
      qface->d = fmax(smallr, qface->d);
    }

    if (faceId == FACE_ZMIN) {
      // Top state at bottom interface
      qface->d = r - HALF_F*drz + sr0*dtdz*HALF_F;
      qface->p = p - HALF_F*dpz + sp0*dtdz*HALF_F;
      qface->u = u - HALF_F*duz + su0*dtdz*HALF_F;
      qface->v = v - HALF_F*dvz + sv0*dtdz*HALF_F;
      qface->w = w - HALF_F*dwz + sw0*dtdz*HALF_F;
      qface->d = fmax(smallr, qface->d);
    }

    if (faceId == FACE_ZMAX) {
      // Top state at bottom interface
      qface->d = r + HALF_F*drz + sr0*dtdz*HALF_F;
      qface->p = p + HALF_F*dpz + sp0*dtdz*HALF_F;
      qface->u = u + HALF_F*duz + su0*dtdz*HALF_F;
      qface->v = v + HALF_F*dvz + sv0*dtdz*HALF_F;
      qface->w = w + HALF_F*dwz + sw0*dtdz*HALF_F;
      qface->d = fmax(smallr, qface->d);
    }

  } // trace_unsplit_3d_along_dir

  /**
   * Compute primitive variables slopes (dqX,dqY,dqZ) for one component
   * from q and its neighbors.
   * This routine is only used in the 3D UNSPLIT integration and 
   * slope_type = 0,1 and 2.
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
    dcen = HALF_F * (qPlusX - qMinusX);
    dsgn = (dcen >= ZERO_F) ? ONE_F : -ONE_F;
    slop = fmin( FABS(dlft), FABS(drgt) );
    dlim = slop;
    if ( (dlft*drgt) <= ZERO_F )
      dlim = ZERO_F;
    *dqX = dsgn * fmin( dlim, FABS(dcen) );
  
    // slopes in second coordinate direction
    dlft = slope_type*(q      - qMinusY);
    drgt = slope_type*(qPlusY - q      );
    dcen = HALF_F * (qPlusY - qMinusY);
    dsgn = (dcen >= ZERO_F) ? ONE_F : -ONE_F;
    slop = fmin( FABS(dlft), FABS(drgt) );
    dlim = slop;
    if ( (dlft*drgt) <= ZERO_F )
      dlim = ZERO_F;
    *dqY = dsgn * fmin( dlim, FABS(dcen) );

    // slopes in third coordinate direction
    dlft = slope_type*(q      - qMinusZ);
    drgt = slope_type*(qPlusZ - q      );
    dcen = HALF_F * (qPlusZ - qMinusZ);
    dsgn = (dcen >= ZERO_F) ? ONE_F : -ONE_F;
    slop = fmin( FABS(dlft), FABS(drgt) );
    dlim = slop;
    if ( (dlft*drgt) <= ZERO_F )
      dlim = ZERO_F;
    *dqZ = dsgn * fmin( dlim, FABS(dcen) );

  } // slope_unsplit_hydro_3d_scalar



  /**
   * Compute primitive variables slope (vector dq) from q and its neighbors.
   * This routine is only used in the 3D UNSPLIT integration and slope_type = 0,1 and 2.
   * 
   * Only slope_type 1 and 2 are supported.
   *
   * \param[in]  q       : current primitive variable state
   * \param[in]  qPlusX  : state in the next neighbor cell along XDIR
   * \param[in]  qMinusX : state in the previous neighbor cell along XDIR
   * \param[in]  qPlusY  : state in the next neighbor cell along YDIR
   * \param[in]  qMinusY : state in the previous neighbor cell along YDIR
   * \param[in]  qPlusZ  : state in the next neighbor cell along ZDIR
   * \param[in]  qMinusZ : state in the previous neighbor cell along ZDIR
   * \param[out] dqX     : reference to an array returning the X slopes
   * \param[out] dqY     : reference to an array returning the Y slopes
   * \param[out] dqZ     : reference to an array returning the Z slopes
   *
   */
  KOKKOS_INLINE_FUNCTION
  void slope_unsplit_hydro_3d(const HydroState *q, 
			      const HydroState *qPlusX, 
			      const HydroState *qMinusX,
			      const HydroState *qPlusY,
			      const HydroState *qMinusY,
			      const HydroState *qPlusZ,
			      const HydroState *qMinusZ,
			      HydroState *dqX,
			      HydroState *dqY,
			      HydroState *dqZ) const
  {
  
    real_t slope_type = params.settings.slope_type;

    if (slope_type==0) {

      dqX->d = ZERO_F;
      dqX->p = ZERO_F;
      dqX->u = ZERO_F;
      dqX->v = ZERO_F;
      dqX->w = ZERO_F;

      dqY->d = ZERO_F;
      dqY->p = ZERO_F;
      dqY->u = ZERO_F;
      dqY->v = ZERO_F;
      dqY->w = ZERO_F;

      dqZ->d = ZERO_F;
      dqZ->p = ZERO_F;
      dqZ->u = ZERO_F;
      dqZ->v = ZERO_F;
      dqZ->w = ZERO_F;

      return;
    }

    if (slope_type==1 || slope_type==2) {  // minmod or average

      slope_unsplit_hydro_3d_scalar( q->d, qPlusX->d, qMinusX->d, qPlusY->d, qMinusY->d, qPlusZ->d, qMinusZ->d,
				     &(dqX->d), &(dqY->d), &(dqZ->d));
      slope_unsplit_hydro_3d_scalar( q->p, qPlusX->p, qMinusX->p, qPlusY->p, qMinusY->p, qPlusZ->p, qMinusZ->p,
				     &(dqX->p), &(dqY->p), &(dqZ->p));
      slope_unsplit_hydro_3d_scalar( q->u, qPlusX->u, qMinusX->u, qPlusY->u, qMinusY->u, qPlusZ->u, qMinusZ->u,
				     &(dqX->u), &(dqY->u), &(dqZ->v));
      slope_unsplit_hydro_3d_scalar( q->v, qPlusX->v, qMinusX->v, qPlusY->v, qMinusY->v, qPlusZ->v, qMinusZ->v,
				     &(dqX->v), &(dqY->v), &(dqZ->v));
      slope_unsplit_hydro_3d_scalar( q->w, qPlusX->w, qMinusX->w, qPlusY->w, qMinusY->w, qPlusZ->w, qMinusZ->w,
				     &(dqX->w), &(dqY->w), &(dqZ->w));

    } // end slope_type == 1 or 2
  
  } // slope_unsplit_hydro_3d

  /**
   * Compute cell fluxes from the Godunov state
   * \param[in]  qgdnv input Godunov state
   * \param[out] flux  output flux vector
   */
  KOKKOS_INLINE_FUNCTION
  void cmpflx(const HydroState *qgdnv, 
	      HydroState *flux) const
  {
    real_t gamma0 = params.settings.gamma0;

    // Compute fluxes
    // Mass density
    flux->d = qgdnv->d * qgdnv->u;
  
    // Normal momentum
    flux->u = flux->d * qgdnv->u + qgdnv->p;
  
    // Transverse momentum
    flux->v = flux->d * qgdnv->v;
    flux->w = flux->d * qgdnv->w;

    // Total energy
    real_t entho = ONE_F / (gamma0 - ONE_F);
    real_t ekin;
    ekin = HALF_F * qgdnv->d * (qgdnv->u*qgdnv->u + qgdnv->v*qgdnv->v + qgdnv->w*qgdnv->w);
  
    real_t etot = qgdnv->p * entho + ekin;
    flux->p = qgdnv->u * (etot + qgdnv->p);

  } // cmpflx
  
  /** 
   * Riemann solver, equivalent to riemann_approx in RAMSES (see file
   * godunov_utils.f90 in RAMSES).
   * 
   * @param[in] qleft  : input left state
   * @param[in] qright : input right state
   * @param[out] qgdnv : output Godunov state
   * @param[out] flux  : output flux
   */
  KOKKOS_INLINE_FUNCTION
  void riemann_approx(const HydroState *qleft,
		      const HydroState *qright,
		      HydroState *qgdnv, 
		      HydroState *flux) const
  {
    real_t gamma0  = params.settings.gamma0;
    real_t gamma6  = params.settings.gamma6;
    real_t smallr  = params.settings.smallr;
    real_t smallc  = params.settings.smallc;
    real_t smallp  = params.settings.smallp;
    real_t smallpp = params.settings.smallpp;

    // Pressure, density and velocity
    real_t rl = fmax(qleft ->d, smallr);
    real_t ul =      qleft ->u;
    real_t pl = fmax(qleft ->p, rl*smallp);
    real_t rr = fmax(qright->d, smallr);
    real_t ur =      qright->u;
    real_t pr = fmax(qright->p, rr*smallp);
  
    // Lagrangian sound speed
    real_t cl = gamma0*pl*rl;
    real_t cr = gamma0*pr*rr;
  
    // First guess
    real_t wl = SQRT(cl);
    real_t wr = SQRT(cr);
    real_t pstar = fmax(((wr*pl+wl*pr)+wl*wr*(ul-ur))/(wl+wr), (real_t) ZERO_F);
    real_t pold = pstar;
    real_t conv = ONE_F;
  
    // Newton-Raphson iterations to find pstar at the required accuracy
    for(int iter = 0; (iter < 10 /*niter_riemann*/) && (conv > 1e-6); ++iter)
      {
	real_t wwl = SQRT(cl*(ONE_F+gamma6*(pold-pl)/pl));
	real_t wwr = SQRT(cr*(ONE_F+gamma6*(pold-pr)/pr));
	real_t ql = 2.0f*wwl*wwl*wwl/(wwl*wwl+cl);
	real_t qr = 2.0f*wwr*wwr*wwr/(wwr*wwr+cr);
	real_t usl = ul-(pold-pl)/wwl;
	real_t usr = ur+(pold-pr)/wwr;
	real_t delp = fmax(qr*ql/(qr+ql)*(usl-usr),-pold);
      
	pold = pold+delp;
	conv = FABS(delp/(pold+smallpp));	 // Convergence indicator
      }
  
    // Star region pressure
    // for a two-shock Riemann problem
    pstar = pold;
    wl = SQRT(cl*(ONE_F+gamma6*(pstar-pl)/pl));
    wr = SQRT(cr*(ONE_F+gamma6*(pstar-pr)/pr));
  
    // Star region velocity
    // for a two shock Riemann problem
    real_t ustar = HALF_F * (ul + (pl-pstar)/wl + ur - (pr-pstar)/wr);
  
    // Left going or right going contact wave
    real_t sgnm = COPYSIGN(ONE_F, ustar);
  
    // Left or right unperturbed state
    real_t ro, uo, po, wo;
    if(sgnm > ZERO_F)
      {
	ro = rl;
	uo = ul;
	po = pl;
	wo = wl;
      }
    else
      {
	ro = rr;
	uo = ur;
	po = pr;
	wo = wr;
      }
    real_t co = fmax(smallc, SQRT(FABS(gamma0*po/ro)));
  
    // Star region density (Shock, fmax prevents vacuum formation in star region)
    real_t rstar = fmax((real_t) (ro/(ONE_F+ro*(po-pstar)/(wo*wo))), (real_t) (smallr));
    // Star region sound speed
    real_t cstar = fmax(smallc, SQRT(FABS(gamma0*pstar/rstar)));
  
    // Compute rarefaction head and tail speed
    real_t spout  = co    - sgnm*uo;
    real_t spin   = cstar - sgnm*ustar;
    // Compute shock speed
    real_t ushock = wo/ro - sgnm*uo;
  
    if(pstar >= po)
      {
	spin  = ushock;
	spout = ushock;
      }
  
    // Sample the solution at x/t=0
    real_t scr = fmax(spout-spin, smallc+FABS(spout+spin));
    real_t frac = HALF_F * (ONE_F + (spout + spin)/scr);

    if (frac != frac) /* Not a Number */
      frac = 0.0;
    else
      frac = frac >= 1.0 ? 1.0 : frac <= 0.0 ? 0.0 : frac;
  
    qgdnv->d = frac*rstar + (ONE_F-frac)*ro;
    qgdnv->u = frac*ustar + (ONE_F-frac)*uo;
    qgdnv->p = frac*pstar + (ONE_F-frac)*po;
  
    if(spout < ZERO_F)
      {
	qgdnv->d = ro;
	qgdnv->u = uo;
	qgdnv->p = po;
      }
  
    if(spin > ZERO_F)
      {
	qgdnv->d = rstar;
	qgdnv->u = ustar;
	qgdnv->p = pstar;
      }
  
    // transverse velocity
    if(sgnm > ZERO_F)
      {
	qgdnv->v = qleft->v;
	qgdnv->w = qleft->w;
      }
    else
      {
	qgdnv->v = qright->v;
	qgdnv->w = qright->w;
      }
  
    cmpflx(qgdnv, flux);
  
  } // riemann_approx

    
  /** 
   * Riemann solver HLLC
   *
   * @param[in] qleft : input left state
   * @param[in] qright : input right state
   * @param[out] qgdnv : output Godunov state
   * @param[out] flux  : output flux
   */
  KOKKOS_INLINE_FUNCTION
  void riemann_hllc(const HydroState *qleft,
		    const HydroState *qright,
		    HydroState *qgdnv,
		    HydroState *flux) const
  {

    real_t gamma0 = params.settings.gamma0;
    real_t smallr = params.settings.smallr;
    real_t smallp = params.settings.smallp;
    real_t smallc = params.settings.smallc;

    const real_t entho = ONE_F / (gamma0 - ONE_F);
  
    // Left variables
    real_t rl = fmax(qleft->d, smallr);
    real_t pl = fmax(qleft->p, rl*smallp);
    real_t ul =      qleft->u;
    
    real_t ecinl = HALF_F*rl*ul*ul;
    ecinl += HALF_F*rl*qleft->v*qleft->v;
    ecinl += HALF_F*rl*qleft->w*qleft->w;

    real_t etotl = pl*entho+ecinl;
    real_t ptotl = pl;

    // Right variables
    real_t rr = fmax(qright->d, smallr);
    real_t pr = fmax(qright->p, rr*smallp);
    real_t ur =      qright->u;

    real_t ecinr = HALF_F*rr*ur*ur;
    ecinr += HALF_F*rr*qright->v*qright->v;
    ecinr += HALF_F*rr*qright->w*qright->w;
  
    real_t etotr = pr*entho+ecinr;
    real_t ptotr = pr;
    
    // Find the largest eigenvalues in the normal direction to the interface
    real_t cfastl = SQRT(fmax(gamma0*pl/rl,smallc*smallc));
    real_t cfastr = SQRT(fmax(gamma0*pr/rr,smallc*smallc));

    // Compute HLL wave speed
    real_t SL = fmin(ul,ur) - fmax(cfastl,cfastr);
    real_t SR = fmax(ul,ur) + fmax(cfastl,cfastr);

    // Compute lagrangian sound speed
    real_t rcl = rl*(ul-SL);
    real_t rcr = rr*(SR-ur);
    
    // Compute acoustic star state
    real_t ustar    = (rcr*ur   +rcl*ul   +  (ptotl-ptotr))/(rcr+rcl);
    real_t ptotstar = (rcr*ptotl+rcl*ptotr+rcl*rcr*(ul-ur))/(rcr+rcl);

    // Left star region variables
    real_t rstarl    = rl*(SL-ul)/(SL-ustar);
    real_t etotstarl = ((SL-ul)*etotl-ptotl*ul+ptotstar*ustar)/(SL-ustar);
    
    // Right star region variables
    real_t rstarr    = rr*(SR-ur)/(SR-ustar);
    real_t etotstarr = ((SR-ur)*etotr-ptotr*ur+ptotstar*ustar)/(SR-ustar);
    
    // Sample the solution at x/t=0
    real_t ro, uo, ptoto, etoto;
    if (SL > ZERO_F) {
      ro=rl;
      uo=ul;
      ptoto=ptotl;
      etoto=etotl;
    } else if (ustar > ZERO_F) {
      ro=rstarl;
      uo=ustar;
      ptoto=ptotstar;
      etoto=etotstarl;
    } else if (SR > ZERO_F) {
      ro=rstarr;
      uo=ustar;
      ptoto=ptotstar;
      etoto=etotstarr;
    } else {
      ro=rr;
      uo=ur;
      ptoto=ptotr;
      etoto=etotr;
    }
      
    // Compute the Godunov flux
    flux->d = ro*uo;
    flux->u = ro*uo*uo+ptoto;
    flux->p = (etoto+ptoto)*uo;
    if (flux->d > ZERO_F) {
      flux->v = flux->d*qleft->v;
      flux->w = flux->d*qleft->w;
    } else {
      flux->v = flux->d*qright->v;
      flux->w = flux->d*qright->w;
    }
  
  } // riemann_hllc

}; // class HydroBaseFunctor3D

} // namespace muscl

} // namespace ppkMHD

#endif // HYDRO_BASE_FUNCTOR_3D_H_
