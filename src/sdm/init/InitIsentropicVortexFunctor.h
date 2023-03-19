#ifndef SDM_INIT_ISENTROPIC_VORTEX_FUNCTOR_H_
#define SDM_INIT_ISENTROPIC_VORTEX_FUNCTOR_H_

#include <limits> // for std::numeric_limits
#ifdef __CUDA_ARCH__
#include <math_constants.h> // for cuda math constants, e.g. CUDART_INF
#endif // __CUDA_ARCH__

#include "shared/kokkos_shared.h"
#include "sdm/SDMBaseFunctor.h"

#include "sdm/SDM_Geometry.h"
#include "sdm/sdm_shared.h" // for DofMap

#include "shared/problems/IsentropicVortexParams.h"

namespace ppkMHD {
namespace sdm {

/*************************************************/
/*************************************************/
/*************************************************/
template<int dim, int N>
class InitIsentropicVortexFunctor : public SDMBaseFunctor<dim,N> {

public:
  using typename SDMBaseFunctor<dim,N>::DataArray;

  static constexpr auto dofMap = DofMap<dim,N>;

  InitIsentropicVortexFunctor(HydroParams            params,
			      SDM_Geometry<dim,N>    sdm_geom,
			      IsentropicVortexParams iparams,
			      DataArray              Udata) :
    SDMBaseFunctor<dim,N>(params,sdm_geom),
    iparams(iparams),
    Udata(Udata) {};

  // static method which does it all: create and execute functor
  static void apply(HydroParams            params,
		    SDM_Geometry<dim,N>    sdm_geom,
		    IsentropicVortexParams iparams,
                    DataArray              Udata)
  {
    std::size_t nbCells = dim == 2 ?
      params.isize*params.jsize : params.isize*params.jsize*params.ksize;

    InitIsentropicVortexFunctor<dim,N> functor(params, sdm_geom, iparams, Udata);
    Kokkos::parallel_for("InitIsentropicVortexFunctor",nbCells, functor);
  }

  /*
   * 2D version.
   */
  //! functor for 2d
  template<int dim_ = dim>
  KOKKOS_INLINE_FUNCTION
  void operator()(const typename std::enable_if<dim_==2, int>::type& index) const
  {

    const int isize = this->params.isize;
    const int jsize = this->params.jsize;
    const int ghostWidth = this->params.ghostWidth;

#ifdef USE_MPI
    const int i_mpi = this->params.myMpiPos[IX];
    const int j_mpi = this->params.myMpiPos[IY];
#else
    const int i_mpi = 0;
    const int j_mpi = 0;
#endif

    const int nx = this->params.nx;
    const int ny = this->params.ny;

    const real_t xmin = this->params.xmin;
    const real_t ymin = this->params.ymin;
    const real_t dx = this->params.dx;
    const real_t dy = this->params.dy;

    const real_t gamma0 = this->params.settings.gamma0;

    // ambient flow
    const real_t rho_a = this->iparams.rho_a;
    //const real_t p_a   = this->iparams.p_a;
    const real_t T_a   = this->iparams.T_a;
    const real_t u_a   = this->iparams.u_a;
    const real_t v_a   = this->iparams.v_a;
    //const real_t w_a   = this->iparams.w_a;

    // vortex center
    real_t vortex_x = this->iparams.vortex_x;
    real_t vortex_y = this->iparams.vortex_y;
    const real_t beta =     this->iparams.beta;

    const bool use_tEnd = this->iparams.use_tEnd;
    if (use_tEnd) {
      const real_t xmax = this->params.xmax;
      const real_t ymax = this->params.ymax;
      vortex_x += this->iparams.tEnd * u_a;
      vortex_y += this->iparams.tEnd * v_a;

      // make sure vortex center is inside the box (periodic boundaries for this test)
      vortex_x = fmod(vortex_x, xmax-xmin);
      vortex_y = fmod(vortex_y, ymax-ymin);
    }

    int i,j;
    index2coord(index,i,j,isize,jsize);

    // loop over cell DoF's
    for (int idy=0; idy<N; ++idy) {
      for (int idx=0; idx<N; ++idx) {

	// lower left corner
	real_t x = xmin + (i+nx*i_mpi-ghostWidth)*dx;
	real_t y = ymin + (j+ny*j_mpi-ghostWidth)*dy;

	// DoF location
	x += this->sdm_geom.solution_pts_1d(idx) * dx;
	y += this->sdm_geom.solution_pts_1d(idy) * dy;

	// relative coordinates versus vortex center
	real_t xp = x - vortex_x;
	real_t yp = y - vortex_y;
	real_t r  = sqrt(xp*xp + yp*yp);

	real_t du = - yp * beta / (2 * M_PI) * exp(0.5*(1.0-r*r));
	real_t dv =   xp * beta / (2 * M_PI) * exp(0.5*(1.0-r*r));

	real_t T = T_a - (gamma0-1)*beta*beta/(8*gamma0*M_PI*M_PI)*exp(1.0-r*r);
	real_t rho = rho_a*pow(T/T_a,1.0/(gamma0-1));

	Udata(i  ,j  , dofMap(idx,idy,0,ID)) = rho;
	Udata(i  ,j  , dofMap(idx,idy,0,IU)) = rho*(u_a + du);
	Udata(i  ,j  , dofMap(idx,idy,0,IV)) = rho*(v_a + dv);
	//Udata(i  ,j  , dofMap(idx,idy,0,IE)) = pow(rho,gamma0)/(gamma0-1.0) +
	Udata(i  ,j  , dofMap(idx,idy,0,IE)) = rho*T/(gamma0-1.0) +
	  0.5*rho*(u_a + du)*(u_a + du) +
	  0.5*rho*(v_a + dv)*(v_a + dv) ;

      } // end for idx
    } // end for idy

  } // end operator () - 2d

  /*
   * 3D version.
   */
  //! functor for 3d
  template<int dim_ = dim>
  KOKKOS_INLINE_FUNCTION
  void operator()(const typename std::enable_if<dim_==3, int>::type& index) const
  {

    const int isize = this->params.isize;
    const int jsize = this->params.jsize;
    const int ksize = this->params.ksize;
    const int ghostWidth = this->params.ghostWidth;

#ifdef USE_MPI
    const int i_mpi = this->params.myMpiPos[IX];
    const int j_mpi = this->params.myMpiPos[IY];
    const int k_mpi = this->params.myMpiPos[IZ];
#else
    const int i_mpi = 0;
    const int j_mpi = 0;
    const int k_mpi = 0;
#endif

    const int nx = this->params.nx;
    const int ny = this->params.ny;
    const int nz = this->params.nz;

    const real_t xmin = this->params.xmin;
    const real_t ymin = this->params.ymin;
    const real_t zmin = this->params.zmin;

    const real_t dx = this->params.dx;
    const real_t dy = this->params.dy;
    const real_t dz = this->params.dz;

    const real_t gamma0 = this->params.settings.gamma0;

    // ambient flow
    const real_t rho_a = this->iparams.rho_a;
    //const real_t p_a   = this->iparams.p_a;
    const real_t T_a   = this->iparams.T_a;
    const real_t u_a   = this->iparams.u_a;
    const real_t v_a   = this->iparams.v_a;
    //const real_t w_a   = this->iparams.w_a;

    // vortex center
    real_t vortex_x = this->iparams.vortex_x;
    real_t vortex_y = this->iparams.vortex_y;
    const real_t beta =     this->iparams.beta;

    const bool use_tEnd = this->iparams.use_tEnd;
    if (use_tEnd) {
      const real_t xmax = this->params.xmax;
      const real_t ymax = this->params.ymax;

      vortex_x += this->iparams.tEnd * u_a;
      vortex_y += this->iparams.tEnd * v_a;

      // make sure vortex center is inside the box (periodic boundaries for this test)
      vortex_x = fmod(vortex_x, xmax-xmin);
      vortex_y = fmod(vortex_y, ymax-ymin);
    }

    // local cell index
    int i,j,k;
    index2coord(index,i,j,k,isize,jsize,ksize);

    // loop over cell DoF's
    for (int idz=0; idz<N; ++idz) {
      for (int idy=0; idy<N; ++idy) {
	for (int idx=0; idx<N; ++idx) {

	  // lower left corner
	  real_t x = xmin + (i+nx*i_mpi-ghostWidth)*dx;
	  real_t y = ymin + (j+ny*j_mpi-ghostWidth)*dy;
	  real_t z = zmin + (k+nz*k_mpi-ghostWidth)*dz;

	  x += this->sdm_geom.solution_pts_1d(idx) * dx;
	  y += this->sdm_geom.solution_pts_1d(idy) * dy;
	  z += this->sdm_geom.solution_pts_1d(idz) * dz;

	  // relative coordinates versus vortex center
	  real_t xp = x - vortex_x;
	  real_t yp = y - vortex_y;
	  real_t r  = sqrt(xp*xp + yp*yp);

	  real_t du = - yp * beta / (2 * M_PI) * exp(0.5*(1.0-r*r));
	  real_t dv =   xp * beta / (2 * M_PI) * exp(0.5*(1.0-r*r));

	  real_t T = T_a - (gamma0-1)*beta*beta/(8*gamma0*M_PI*M_PI)*exp(1.0-r*r);
	  real_t rho = rho_a*pow(T/T_a,1.0/(gamma0-1));

	  Udata(i,j,k, dofMap(idx,idy,idz,ID)) = rho;
	  Udata(i,j,k, dofMap(idx,idy,idz,IU)) = rho*(u_a + du);
	  Udata(i,j,k, dofMap(idx,idy,idz,IV)) = rho*(v_a + dv);
	  Udata(i,j,k, dofMap(idx,idy,idz,IW)) = 0.0;
	  Udata(i,j,k, dofMap(idx,idy,idz,IE)) = rho*T/(gamma0-1.0) +
	    0.5*rho*(u_a + du)*(u_a + du) +
	    0.5*rho*(v_a + dv)*(v_a + dv) ;

	} // end for idx
      } // end for idy
    } // end for idz

  } // end operator () - 3d

  IsentropicVortexParams iparams;
  DataArray Udata;

}; // InitIsentropicVortexFunctor

} // namespace sdm
} // namespace ppkMHD

#endif // SDM_INIT_ISENTROPIC_VORTEX_FUNCTOR_H_
