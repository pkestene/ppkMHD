/**
 * \file SDM_Positivity_preserving.h
 *
 * Implement ideas from article
 * "On positivity-preserving high order discontinuous Galerkin schemes for
 * compressible Euler equations on rectangular meshes", Xiangxiong Zhang, 
 * Chi-Wang Shu, Journal of Computational Physics Volume 229, Issue 23,
 * 20 November 2010, Pages 8918-8934
 *
 * The idea is to ensure/preserve positivity after each Runge-Kutta stage.
 */
#ifndef SDM_POSITIVITY_PRESERVING_H_
#define SDM_POSITIVITY_PRESERVING_H_

#include <limits> // for std::numeric_limits
#ifdef __CUDA_ARCH__
#include <math_constants.h> // for cuda math constants, e.g. CUDART_INF
#endif // __CUDA_ARCH__

#include "shared/kokkos_shared.h"
#include "sdm/SDMBaseFunctor.h"

#include "sdm/SDM_Geometry.h"
#include "sdm/sdm_shared.h" // for DofMap

#include "shared/EulerEquations.h"

namespace sdm {

/*************************************************/
/*************************************************/
/*************************************************/
/**
 *
 */
template<int dim, int N>
class Apply_positivity_at_solution_points_Functor : public SDMBaseFunctor<dim,N> {

public:
  using typename SDMBaseFunctor<dim,N>::DataArray;
  using typename SDMBaseFunctor<dim,N>::solution_values_t;

  static constexpr auto dofMapS = DofMap<dim,N>;

  Apply_positivity_at_solution_points_Functor(HydroParams         params,
					      SDM_Geometry<dim,N> sdm_geom,
					      DataArray           UdataSol,
					      DataArray           Uaverage) :
    SDMBaseFunctor<dim,N>(params,sdm_geom),
    UdataSol(UdataSol),
    Uaverage(Uaverage)
  {};

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
    
    const int nbvar = this->params.nbvar;
    
    // local cell index
    int i,j;
    index2coord(index,i,j,isize,jsize);

    // average density
    real_t rho_ave = Uaverage(i,j,ID);
    
    // density positivity
    real_t rho_min = 1e20; // something big
    for (int idy=0; idy<N; ++idy) {
      for (int idx=0; idx<N; ++idx) {

	const real_t rho = UdataSol(i,j,dofMapS(idx,idy,0,ID));
	rho_min = rho_min < rho ? rho_min : rho;
	
      } // end for idx
    } // end for idy

    const real_t eps = this->params.settings.smallr; // a small density
    const real_t ratio = (rho_ave - eps)/(rho_ave - rho_min) + 1e-13;
    const real_t theta1 = ratio < 1.0 ? ratio : 1.0;

    // check if we need to modify density
    if (theta1 < 1.0) {

    }
    
  } // end operator() - 2d
    
  //! solution data array
  DataArray UdataSol;
  DataArray Uaverage;
  
}; // class Apply_positivity_at_solution_points_Functor

} // namespace

#endif // SDM_POSITIVITY_PRESERVING_H_
