/**
 * All possible Riemann solvers or so.
 */
#ifndef RIEMANN_SOLVERS_H_
#define RIEMANN_SOLVERS_H_

#include <math.h>

#include "HydroParams.h"
#include "HydroState.h"

namespace ppkMHD {
  
/**
 * Compute cell fluxes from the Godunov state
 * \param[in]  qgdnv input Godunov state
 * \param[out] flux  output flux vector
 */
KOKKOS_INLINE_FUNCTION
void cmpflx(const HydroState2d *qgdnv, 
	    HydroState2d *flux,
	    const HydroParams& params)
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
  ekin = HALF_F * qgdnv->d * (qgdnv->u*qgdnv->u + qgdnv->v*qgdnv->v);
  
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
//template<DimensionType>
KOKKOS_INLINE_FUNCTION
void riemann_approx(const HydroState2d *qleft,
		    const HydroState2d *qright,
		    HydroState2d *qgdnv, 
		    HydroState2d *flux,
		    const HydroParams& params)
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
    }
  else
    {
      qgdnv->v = qright->v;
    }
  
  cmpflx(qgdnv, flux, params);
  
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
void riemann_hllc(const HydroState2d *qleft,
		  const HydroState2d *qright,
		  HydroState2d *qgdnv,
		  HydroState2d *flux,
		  const HydroParams& params)
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

  real_t etotl = pl*entho+ecinl;
  real_t ptotl = pl;

  // Right variables
  real_t rr = fmax(qright->d, smallr);
  real_t pr = fmax(qright->p, rr*smallp);
  real_t ur =      qright->u;

  real_t ecinr = HALF_F*rr*ur*ur;
  ecinr += HALF_F*rr*qright->v*qright->v;
  
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
  } else {
    flux->v = flux->d*qright->v;
  }
  
} // riemann_hllc

/**
 * Wrapper function calling the actual riemann solver.
 */
KOKKOS_INLINE_FUNCTION
void riemann_hydro(const HydroState2d *qleft,
		   const HydroState2d *qright,
		   HydroState2d *qgdnv, 
		   HydroState2d *flux,
		   const HydroParams& params)
{

  if        (params.riemannSolverType == RIEMANN_APPROX) {
    
    riemann_approx(qleft,qright,qgdnv,flux,params);
    
  } else if (params.riemannSolverType == RIEMANN_HLLC) {
    
    riemann_hllc  (qleft,qright,qgdnv,flux,params);

  }
  
} // riemann_hydro

} // namespace ppkMHD

#endif // RIEMANN_SOLVERS_H_
