#ifndef SDM_INIT_GRESHO_VORTEX_FUNCTOR_H_
#define SDM_INIT_GRESHO_VORTEX_FUNCTOR_H_

#include <limits> // for std::numeric_limits
#ifdef __CUDA_ARCH__
#include <math_constants.h> // for cuda math constants, e.g. CUDART_INF
#endif // __CUDA_ARCH__

#include "shared/kokkos_shared.h"
#include "sdm/SDMBaseFunctor.h"

#include "sdm/SDM_Geometry.h"
#include "sdm/sdm_shared.h" // for DofMap

#include "shared/problems/GreshoParams.h"

namespace sdm {

/*************************************************/
/*************************************************/
/*************************************************/
template<int dim, int N>
class InitGreshoVortexFunctor : public SDMBaseFunctor<dim,N> {

public:
  using typename SDMBaseFunctor<dim,N>::DataArray;

  static constexpr auto dofMap = DofMap<dim,N>;

  InitGreshoVortexFunctor(HydroParams         params,
			  SDM_Geometry<dim,N> sdm_geom,
			  GreshoParams        gvParams,
			  DataArray           Udata) :
    SDMBaseFunctor<dim,N>(params,sdm_geom),
    gvParams(gvParams),
    Udata(Udata) {};
  
  // static method which does it all: create and execute functor
  static void apply(HydroParams         params,
                    SDM_Geometry<dim,N> sdm_geom,
                    GreshoParams        gvParams,
                    DataArray           Udata)
  {
    int nbCells = dim==2 ? 
      params.isize*params.jsize : 
      params.isize*params.jsize*params.ksize;
    
    InitGreshoVortexFunctor functor(params, sdm_geom, gvParams, Udata);
    Kokkos::parallel_for("InitGreshoVortexFunctor",nbCells, functor);
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

    // gresho vortex problem parameters
    const real_t rho0  = gvParams.rho0;
    const real_t Ma    = gvParams.Ma;

    const real_t p0 = rho0 / (gamma0 * Ma * Ma);
    
    // local cell index
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

	// polar coordinate
	real_t r = sqrt(x*x+y*y);
	real_t theta = atan2(y,x);
	
	real_t cosT = cos(theta);
	real_t sinT = sin(theta);
	
	real_t uphi, p;

	if ( r < 0.2 ) {
	  
	  uphi = 5*r;
	  p    = p0 + 25/2.0*r*r;
	  
	} else if ( r < 0.4 ) {
	  
	  uphi = 2 - 5 * r;
	  p    = p0 + 25/2.0*r*r + 4*(1-5*r-log(0.2)+log(r));
	  
	} else {
	  
	  uphi = 0;
	  p    = p0-2+4*log(2.0);
	  
	}
	
	Udata(i,j,dofMap(idx,idy,0,ID)) = rho0;
	Udata(i,j,dofMap(idx,idy,0,IU)) = rho0 * (-sinT * uphi);
	Udata(i,j,dofMap(idx,idy,0,IV)) = rho0 * ( cosT * uphi);
	Udata(i,j,dofMap(idx,idy,0,IE)) = p/(gamma0-1.0) +
	  0.5*(Udata(i,j,dofMap(idx,idy,0,IU))*Udata(i,j,dofMap(idx,idy,0,IU)) +
	       Udata(i,j,dofMap(idx,idy,0,IV))*Udata(i,j,dofMap(idx,idy,0,IV)))/
	  Udata(i,j,dofMap(idx,idy,0,ID));
	
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

    // gresho problem parameters
    const real_t rho0  = gvParams.rho0;
    const real_t Ma    = gvParams.Ma;

    const real_t p0 = rho0 / (gamma0 * Ma * Ma);

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

	  // polar coordinate
	  real_t r = sqrt(x*x+y*y);
	  real_t theta = atan2(y,x);
	  
	  real_t cosT = cos(theta);
	  real_t sinT = sin(theta);

	  real_t uphi, p;

	  if ( r < 0.2 ) {
	    
	    uphi = 5*r;
	    p    = p0 + 25/2.0*r*r;
	    
	  } else if ( r < 0.4 ) {
	    
	  uphi = 2 - 5 * r;
	  p    = p0 + 25/2.0*r*r + 4*(1-5*r-log(0.2)+log(r));
	  
	  } else {
	    
	    uphi = 0;
	    p    = p0-2+4*log(2.0);
	    
	  }
	  
	  Udata(i,j,k,dofMap(idx,idy,idz,ID)) = rho0;
	  Udata(i,j,k,dofMap(idx,idy,idz,IU)) = rho0 * (-sinT * uphi);
	  Udata(i,j,k,dofMap(idx,idy,idz,IV)) = rho0 * ( cosT * uphi);
	  Udata(i,j,k,dofMap(idx,idy,idz,IW)) = 0.0;
	  Udata(i,j,k,dofMap(idx,idy,idz,IE)) = p/(gamma0-1.0) +
	    0.5*(Udata(i,j,k,dofMap(idx,idy,idz,IU))*Udata(i,j,k,dofMap(idx,idy,idz,IU)) +
		 Udata(i,j,k,dofMap(idx,idy,idz,IV))*Udata(i,j,k,dofMap(idx,idy,idz,IV)) +
		 Udata(i,j,k,dofMap(idx,idy,idz,IW))*Udata(i,j,k,dofMap(idx,idy,idz,IW)))/
	    Udata(i,j,k,dofMap(idx,idy,idz,ID));
	  
	} // end for idx
      } // end for idy
    } // end for idz
    
  } // end operator () - 3d
  
  GreshoParams gvParams;
  DataArray    Udata;
  
}; // InitGreshoVortexFunctor

} // namespace sdm

#endif // SDM_INIT_GRESHO_VORTEX_FUNCTOR_H_
