#ifndef SDM_INIT_IMPLODE_FUNCTOR_H_
#define SDM_INIT_IMPLODE_FUNCTOR_H_

#include <limits> // for std::numeric_limits
#ifdef __CUDA_ARCH__
#include <math_constants.h> // for cuda math constants, e.g. CUDART_INF
#endif // __CUDA_ARCH__

#include "shared/kokkos_shared.h"
#include "sdm/SDMBaseFunctor.h"

#include "sdm/SDM_Geometry.h"
#include "sdm/sdm_shared.h" // for DofMap

#include "shared/problems/ImplodeParams.h"

namespace sdm {

/*************************************************/
/*************************************************/
/*************************************************/
template<int dim, int N>
class InitImplodeFunctor : public SDMBaseFunctor<dim,N> {

public:
  using typename SDMBaseFunctor<dim,N>::DataArray;

  static constexpr auto dofMap = DofMap<dim,N>;
  
  InitImplodeFunctor(HydroParams         params,
		     SDM_Geometry<dim,N> sdm_geom,
		     ImplodeParams       iparams,
		     DataArray           Udata) :
    SDMBaseFunctor<dim,N>(params,sdm_geom),
    iparams(iparams),
    Udata(Udata) {};

  // static method which does it all: create and execute functor
  static void apply(HydroParams         params,
                    SDM_Geometry<dim,N> sdm_geom,
                    ImplodeParams       iparams,
                    DataArray           Udata)
  {
    int nbCells = dim==2 ? 
      params.isize*params.jsize : 
      params.isize*params.jsize*params.ksize;
    
    InitImplodeFunctor functor(params, sdm_geom, iparams, Udata);
    Kokkos::parallel_for("InitImplodeFunctor",nbCells, functor);
  }

  /*
   * 2D version.
   */
  //! functor for 2d 
  template<int dim_ = dim>
  KOKKOS_INLINE_FUNCTION
  void operator()(const typename Kokkos::Impl::enable_if<dim_==2, int>::type& index) const
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
    
    const real_t xmax = this->params.xmax;
    //const real_t ymax = this->params.ymax;
    
    const real_t dx = this->params.dx;
    const real_t dy = this->params.dy;
    
    const real_t gamma0 = this->params.settings.gamma0;

    // outer parameters
    const real_t rho_out = this->iparams.rho_out;
    const real_t p_out   = this->iparams.p_out;
    const real_t u_out   = this->iparams.u_out;
    const real_t v_out   = this->iparams.v_out;

    // inner parameters
    const real_t rho_in  = this->iparams.rho_in;
    const real_t p_in    = this->iparams.p_in;
    const real_t u_in    = this->iparams.u_in;
    const real_t v_in    = this->iparams.v_in;

    // local cell index
    int i,j;
    index2coord(index,i,j,isize,jsize);

    // loop over cell DoF's
    for (int idy=0; idy<N; ++idy) {
      for (int idx=0; idx<N; ++idx) {

	// lower left corner
	real_t x = xmin + (i+nx*i_mpi-ghostWidth)*dx;
	real_t y = ymin + (j+ny*j_mpi-ghostWidth)*dy;

	x += this->sdm_geom.solution_pts_1d(idx) * dx;
	y += this->sdm_geom.solution_pts_1d(idy) * dy;
	
	bool tmp;
	if (this->iparams.shape == 1)
	  tmp = x+y*y > 0.5 && x+y*y < 1.5;
	else
	  tmp = x+y > (xmin+xmax)/2. + ymin;
	
	if (tmp) {
	  Udata(i  ,j  , dofMap(idx,idy,0,ID)) = rho_out;
	  Udata(i  ,j  , dofMap(idx,idy,0,IE)) = p_out/(gamma0-1.0) + 0.5 * rho_out * (u_out*u_out + v_out*v_out);
	  Udata(i  ,j  , dofMap(idx,idy,0,IU)) = u_out;
	  Udata(i  ,j  , dofMap(idx,idy,0,IV)) = v_out;
	} else {
	  Udata(i  ,j  , dofMap(idx,idy,0,ID)) = rho_in;
	  Udata(i  ,j  , dofMap(idx,idy,0,IE)) = p_in/(gamma0-1.0) + 0.5 * rho_in * (u_in*u_in + v_in*v_in);
	  Udata(i  ,j  , dofMap(idx,idy,0,IU)) = u_in;
	  Udata(i  ,j  , dofMap(idx,idy,0,IV)) = v_in;
	}
	
      } // end for idx
    } // end for idy
    
  } // end operator () - 2d

  /*
   * 3D version.
   */
  //! functor for 3d 
  template<int dim_ = dim>
  KOKKOS_INLINE_FUNCTION
  void operator()(const typename Kokkos::Impl::enable_if<dim_==3, int>::type& index) const
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

    const real_t xmax = this->params.xmax;
    //const real_t ymax = this->params.ymax;
    //const real_t zmax = this->params.zmax;

    const real_t dx = this->params.dx;
    const real_t dy = this->params.dy;
    const real_t dz = this->params.dz;
    
    const real_t gamma0 = this->params.settings.gamma0;

    // outer parameters
    const real_t rho_out = this->iparams.rho_out;
    const real_t p_out   = this->iparams.p_out;
    const real_t u_out   = this->iparams.u_out;
    const real_t v_out   = this->iparams.v_out;
    const real_t w_out   = this->iparams.w_out;

    // inner parameters
    const real_t rho_in  = this->iparams.rho_in;
    const real_t p_in    = this->iparams.p_in;
    const real_t u_in    = this->iparams.u_in;
    const real_t v_in    = this->iparams.v_in;
    const real_t w_in    = this->iparams.w_in;

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
	  
	  bool tmp;
	  if (this->iparams.shape == 1)
	    tmp = x+y+z > 0.5 && x+y+z < 2.5;
	  else
	    tmp = x+y+z > (xmin+xmax)/2. + ymin + zmin;
	  
	  if (tmp) {
	    Udata(i  ,j  ,k  , dofMap(idx,idy,idz,ID)) = rho_out;
	    Udata(i  ,j  ,k  , dofMap(idx,idy,idz,IE)) = p_out/(gamma0-1.0) + 0.5 * rho_out *
	(u_out*u_out + v_out*v_out + w_out*w_out);
	    Udata(i  ,j  ,k  , dofMap(idx,idy,idz,IU)) = u_out;
	    Udata(i  ,j  ,k  , dofMap(idx,idy,idz,IV)) = v_out;
	    Udata(i  ,j  ,k  , dofMap(idx,idy,idz,IW)) = w_out;
	  } else {
	    Udata(i  ,j  ,k  , dofMap(idx,idy,idz,ID)) = rho_in;
	    Udata(i  ,j  ,k  , dofMap(idx,idy,idz,IE)) = p_in/(gamma0-1.0) + 0.5 * rho_in *
	(u_in*u_in + v_in*v_in + w_in*w_in);
	    Udata(i  ,j  ,k  , dofMap(idx,idy,idz,IU)) = u_in;
	    Udata(i  ,j  ,k  , dofMap(idx,idy,idz,IV)) = v_in;
	    Udata(i  ,j  ,k  , dofMap(idx,idy,idz,IW)) = w_in;
	  }
	  
	} // end for idx
      } // end for idy
    } // end for idz
    
  } // end operator () - 3d
  
  ImplodeParams iparams;
  DataArray     Udata;

}; // InitImplodeFunctor

} // namespace sdm

#endif // SDM_INIT_IMPLODE_FUNCTOR_H_

