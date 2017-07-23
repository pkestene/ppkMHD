#ifndef SDM_LIMITER_FUNCTORS_H_
#define SDM_LIMITER_FUNCTORS_H_

#include <limits> // for std::numeric_limits
#ifdef __CUDA_ARCH__
#include <math_constants.h> // for cuda math constants, e.g. CUDART_INF
#endif // __CUDA_ARCH__

#include "shared/kokkos_shared.h"
#include "sdm/SDMBaseFunctor.h"

#include "sdm/SDM_Geometry.h"
#include "sdm/sdm_shared.h" // for DofMap

namespace sdm {

/*************************************************/
/*************************************************/
/*************************************************/
/**
 * This functor computes the average HydroState in each cell
 * and store the result in Uaverage.
 */
template<int dim, int N>
class Average_Conservative_Variables_Functor : public SDMBaseFunctor<dim,N> {

public:
  using typename SDMBaseFunctor<dim,N>::DataArray;
  using typename SDMBaseFunctor<dim,N>::HydroState;
  
  static constexpr auto dofMap = DofMap<dim,N>;

  Average_Conservative_Variables_Functor(HydroParams         params,
					 SDM_Geometry<dim,N> sdm_geom,
					 DataArray           Udata,
					 DataArray           Uaverage) :
    SDMBaseFunctor<dim,N>(params,sdm_geom),
    Udata(Udata),
    Uaverage(Uaverage)
  {};

  // ================================================
  //
  // 2D version.
  //
  // ================================================
  //! functor for 2d 
  template<int dim_ = dim>
  KOKKOS_INLINE_FUNCTION
  void operator()(const typename Kokkos::Impl::enable_if<dim_==2, int>::type& index) const
  {
    const int isize = this->params.isize;
    const int jsize = this->params.jsize;

    const int nbvar = this->params.nbvar;

    // local cell index
    int i,j;
    index2coord(index,i,j,isize,jsize);

    // for each variables
    for (int ivar = 0; ivar<nbvar; ++ivar) {

      real_t tmp = 0.0;

      // perform the Gauss-Chebyshev quadrature
      
      // for each DoFs
      for (int idy=0; idy<N; ++idy) {
	real_t y = this->sdm_geom.solution_pts_1d_host(idy);
	real_t wy = sqrt(y-y*y);
	
      	for (int idx=0; idx<N; ++idx) {
	  real_t x = this->sdm_geom.solution_pts_1d_host(idx);
	  real_t wx = sqrt(x-x*x);
	  
	  tmp += Udata(i,j, dofMap(idx,idy,0,ivar)) * wx * wy;
	  
	} // for idx
      } // for idy

      // final scaling
      tmp *= (M_PI/N)*(M_PI/N);

      Uaverage(i,j,ivar) = tmp;
      
    } // end for ivar
    
  } // operator () - 2d

  // ================================================
  //
  // 3D version.
  //
  // ================================================
  //! functor for 3d 
  template<int dim_ = dim>
  KOKKOS_INLINE_FUNCTION
  void operator()(const typename Kokkos::Impl::enable_if<dim_==3, int>::type& index) const
  {

    const int isize = this->params.isize;
    const int jsize = this->params.jsize;
    const int ksize = this->params.ksize;

    const int nbvar = this->params.nbvar;

    // local cell index
    int i,j,k;
    index2coord(index,i,j,k,isize,jsize,ksize);

    // for each variables
    for (int ivar = 0; ivar<nbvar; ++ivar) {

      real_t tmp = 0.0;

      // perform the Gauss-Chebyshev quadrature
      
      // for each DoFs
      for (int idz=0; idz<N; ++idz) {
	real_t z = this->sdm_geom.solution_pts_1d_host(idz);
	real_t wz = sqrt(z-z*z);
	
	for (int idy=0; idy<N; ++idy) {
	  real_t y = this->sdm_geom.solution_pts_1d_host(idy);
	  real_t wy = sqrt(y-y*y);
	  
	  for (int idx=0; idx<N; ++idx) {
	    real_t x = this->sdm_geom.solution_pts_1d_host(idx);
	    real_t wx = sqrt(x-x*x);
	    
	    tmp += Udata(i,j,k, dofMap(idx,idy,idz,ivar)) * wx * wy * wz;
	  
	  } // for idx
	} // for idy
      } // for idz
      
      // final scaling
      tmp *= (M_PI/N)*(M_PI/N)*(M_PI/N);

      Uaverage(i,j,k,ivar) = tmp;
      
    } // end for ivar
    
  } // operator () - 3d
  
  DataArray Udata;
  DataArray Uaverage;

}; // class Average_Conservative_Variables_Functor

} // namespace sdm

#endif // SDM_LIMITER_FUNCTORS_H_
