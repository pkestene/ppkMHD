#ifndef SDM_RUN_FUNCTORS_H_
#define SDM_RUN_FUNCTORS_H_

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
template<int dim, int N>
class SDM_Erase_Functor : public SDMBaseFunctor<dim,N> {

public:
  using typename SDMBaseFunctor<dim,N>::DataArray;

  static constexpr auto dofMap = DofMap<dim,N>;
  
  SDM_Erase_Functor(HydroParams         params,
		    SDM_Geometry<dim,N> sdm_geom,
		    DataArray           Udata) :
    SDMBaseFunctor<dim,N>(params,sdm_geom), Udata(Udata) {};

  
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

    // local cell index
    int i,j;
    index2coord(index,i,j,isize,jsize);

    // loop over cell DoF's
    for (int idy=0; idy<N; ++idy) {
      for (int idx=0; idx<N; ++idx) {
	
	Udata(i  ,j  , dofMap(idx,idy,0,ID)) = 0.0;
	Udata(i  ,j  , dofMap(idx,idy,0,IP)) = 0.0;
	Udata(i  ,j  , dofMap(idx,idy,0,IU)) = 0.0;
	Udata(i  ,j  , dofMap(idx,idy,0,IV)) = 0.0;
	
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
    
    // local cell index
    int i,j,k;
    index2coord(index,i,j,k,isize,jsize,ksize);

    // loop over cell DoF's
    for (int idz=0; idz<N; ++idz) {
      for (int idy=0; idy<N; ++idy) {
	for (int idx=0; idx<N; ++idx) {
	  
	  Udata(i  ,j  ,k  , dofMap(idx,idy,idz,ID)) = 0.0;
	  Udata(i  ,j  ,k  , dofMap(idx,idy,idz,IP)) = 0.0;
	  Udata(i  ,j  ,k  , dofMap(idx,idy,idz,IU)) = 0.0;
	  Udata(i  ,j  ,k  , dofMap(idx,idy,idz,IV)) = 0.0;
	  Udata(i  ,j  ,k  , dofMap(idx,idy,idz,IW)) = 0.0;
	  
	} // end for idx
      } // end for idy
    } // end for idz
    
  } // end operator () - 3d
  
  DataArray Udata;

}; // SDM_Erase_Functor

} // namespace sdm

#endif // SDM_RUN_FUNCTORS_H_

