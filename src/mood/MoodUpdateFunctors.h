#ifndef MOOD_UPDATE_FUNCTORS_H_
#define MOOD_UPDATE_FUNCTORS_H_

#include "shared/kokkos_shared.h"
#include "shared/HydroParams.h"
#include "shared/HydroState.h"

#include "mood/MoodBaseFunctor.h"

namespace mood {

// =======================================================================
// =======================================================================
/**
 * Given the fluxes array, perform update of Udata (conservative variable
 * array).
 *
 * \tparam dim dimension (2 or 3).
 */
template<int dim>
class UpdateFunctor
{

public:
  //! Decide at compile-time which data array to use
  using DataArray  = typename std::conditional<dim==2,DataArray2d,DataArray3d>::type;

  UpdateFunctor(HydroParams params,
		DataArray Udata,
		DataArray FluxData_x,
		DataArray FluxData_y,
		DataArray FluxData_z) :
    params(params),
    Udata(Udata),
    FluxData_x(FluxData_x),
    FluxData_y(FluxData_y),
    FluxData_z(FluxData_z)
  {};
  
  //! functor for 2d 
  template<int dim_ = dim>
  KOKKOS_INLINE_FUNCTION
  void operator()(const typename Kokkos::Impl::enable_if<dim_==2, int>::type& index)  const
  {
    const int isize = params.isize;
    const int jsize = params.jsize;
    const int ghostWidth = params.ghostWidth;
    
    int i,j;
    index2coord(index,i,j,isize,jsize);

    if(j >= ghostWidth && j < jsize-ghostWidth  &&
       i >= ghostWidth && i < isize-ghostWidth ) {

      Udata(i  ,j  , ID) +=  FluxData_x(i  ,j  , ID);
      Udata(i  ,j  , IP) +=  FluxData_x(i  ,j  , IP);
      Udata(i  ,j  , IU) +=  FluxData_x(i  ,j  , IU);
      Udata(i  ,j  , IV) +=  FluxData_x(i  ,j  , IV);

      Udata(i  ,j  , ID) -=  FluxData_x(i+1,j  , ID);
      Udata(i  ,j  , IP) -=  FluxData_x(i+1,j  , IP);
      Udata(i  ,j  , IU) -=  FluxData_x(i+1,j  , IU);
      Udata(i  ,j  , IV) -=  FluxData_x(i+1,j  , IV);
      
      Udata(i  ,j  , ID) +=  FluxData_y(i  ,j  , ID);
      Udata(i  ,j  , IP) +=  FluxData_y(i  ,j  , IP);
      Udata(i  ,j  , IU) +=  FluxData_y(i  ,j  , IU);
      Udata(i  ,j  , IV) +=  FluxData_y(i  ,j  , IV);
      
      Udata(i  ,j  , ID) -=  FluxData_y(i  ,j+1, ID);
      Udata(i  ,j  , IP) -=  FluxData_y(i  ,j+1, IP);
      Udata(i  ,j  , IU) -=  FluxData_y(i  ,j+1, IU);
      Udata(i  ,j  , IV) -=  FluxData_y(i  ,j+1, IV);

    } // end if
    
  } // end operator ()
  
  //! functor for 3d 
  template<int dim_ = dim>
  KOKKOS_INLINE_FUNCTION
  void operator()(const typename Kokkos::Impl::enable_if<dim_==3, int>::type& index)  const
  {
    const int isize = params.isize;
    const int jsize = params.jsize;
    const int ksize = params.ksize;
    const int ghostWidth = params.ghostWidth;
    
    int i,j,k;
    index2coord(index,i,j,k,isize,jsize,ksize);

    if(k >= ghostWidth && k < ksize-ghostWidth  &&
       j >= ghostWidth && j < jsize-ghostWidth  &&
       i >= ghostWidth && i < isize-ghostWidth ) {

      Udata(i  ,j  ,k  , ID) +=  FluxData_x(i  ,j  ,k  , ID);
      Udata(i  ,j  ,k  , IP) +=  FluxData_x(i  ,j  ,k  , IP);
      Udata(i  ,j  ,k  , IU) +=  FluxData_x(i  ,j  ,k  , IU);
      Udata(i  ,j  ,k  , IV) +=  FluxData_x(i  ,j  ,k  , IV);
      Udata(i  ,j  ,k  , IW) +=  FluxData_x(i  ,j  ,k  , IW);

      Udata(i  ,j  ,k  , ID) -=  FluxData_x(i+1,j  ,k  , ID);
      Udata(i  ,j  ,k  , IP) -=  FluxData_x(i+1,j  ,k  , IP);
      Udata(i  ,j  ,k  , IU) -=  FluxData_x(i+1,j  ,k  , IU);
      Udata(i  ,j  ,k  , IV) -=  FluxData_x(i+1,j  ,k  , IV);
      Udata(i  ,j  ,k  , IW) -=  FluxData_x(i+1,j  ,k  , IW);
      
      Udata(i  ,j  ,k  , ID) +=  FluxData_y(i  ,j  ,k  , ID);
      Udata(i  ,j  ,k  , IP) +=  FluxData_y(i  ,j  ,k  , IP);
      Udata(i  ,j  ,k  , IU) +=  FluxData_y(i  ,j  ,k  , IU);
      Udata(i  ,j  ,k  , IV) +=  FluxData_y(i  ,j  ,k  , IV);
      Udata(i  ,j  ,k  , IW) +=  FluxData_y(i  ,j  ,k  , IW);
      
      Udata(i  ,j  ,k  , ID) -=  FluxData_y(i  ,j+1,k  , ID);
      Udata(i  ,j  ,k  , IP) -=  FluxData_y(i  ,j+1,k  , IP);
      Udata(i  ,j  ,k  , IU) -=  FluxData_y(i  ,j+1,k  , IU);
      Udata(i  ,j  ,k  , IV) -=  FluxData_y(i  ,j+1,k  , IV);
      Udata(i  ,j  ,k  , IW) -=  FluxData_y(i  ,j+1,k  , IW);

      Udata(i  ,j  ,k  , ID) +=  FluxData_z(i  ,j  ,k  , ID);
      Udata(i  ,j  ,k  , IP) +=  FluxData_z(i  ,j  ,k  , IP);
      Udata(i  ,j  ,k  , IU) +=  FluxData_z(i  ,j  ,k  , IU);
      Udata(i  ,j  ,k  , IV) +=  FluxData_z(i  ,j  ,k  , IV);
      Udata(i  ,j  ,k  , IW) +=  FluxData_z(i  ,j  ,k  , IW);

      Udata(i  ,j  ,k  , ID) -=  FluxData_z(i  ,j  ,k+1, ID);
      Udata(i  ,j  ,k  , IP) -=  FluxData_z(i  ,j  ,k+1, IP);
      Udata(i  ,j  ,k  , IU) -=  FluxData_z(i  ,j  ,k+1, IU);
      Udata(i  ,j  ,k  , IV) -=  FluxData_z(i  ,j  ,k+1, IV);
      Udata(i  ,j  ,k  , IW) -=  FluxData_z(i  ,j  ,k+1, IW);

    } // end if
    
  } // end operator ()
  
  HydroParams params;
  DataArray   Udata;
  DataArray   FluxData_x;
  DataArray   FluxData_y;
  DataArray   FluxData_z;
  
}; // UpdateFunctor

// =======================================================================
// =======================================================================
/**
 * This functor tries to perform update on density, if density or 
 * pressure becomes negative, we flag the cells for recompute.
 *
 * We use MoodBasefunctor as base class to inherit method to compute 
 * primitive variables.
 */
template<int dim,
	 int degree>
class ComputeMoodFlagsUpdateFunctor : public MoodBaseFunctor<dim,degree>
{

public:
  using typename MoodBaseFunctor<dim,degree>::DataArray;
  using typename MoodBaseFunctor<dim,degree>::HydroState;

  ComputeMoodFlagsUpdateFunctor(HydroParams params,
				DataArray Udata,
				DataArray Flags,
				DataArray FluxData_x,
				DataArray FluxData_y,
				DataArray FluxData_z) :
    MoodBaseFunctor<dim,degree>(params),
    Udata(Udata),
    Flags(Flags),
    FluxData_x(FluxData_x),
    FluxData_y(FluxData_y),
    FluxData_z(FluxData_z)
  {};
  
  //! functor for 2d 
  template<int dim_ = dim>
  KOKKOS_INLINE_FUNCTION
  void operator()(const typename Kokkos::Impl::enable_if<dim_==2, int>::type& index) const
  {
    const int isize = this->params.isize;
    const int jsize = this->params.jsize;
    const int ghostWidth = this->params.ghostWidth;
    
    int i,j;
    index2coord(index,i,j,isize,jsize);

    // set flags to zero
    Flags(i,j,0) = 0.0;

    real_t flag_tmp = 0.0;
    
    if(j >= ghostWidth && j < jsize-ghostWidth  &&
       i >= ghostWidth && i < isize-ghostWidth ) {

      real_t rho = Udata(i  ,j  , ID);
      real_t e   = Udata(i  ,j  , IP);
      real_t u   = Udata(i  ,j  , IU);
      real_t v   = Udata(i  ,j  , IV);
      
      real_t rho_new = rho
	+ FluxData_x(i  ,j  , ID)
	- FluxData_x(i+1,j  , ID)
	+ FluxData_y(i  ,j  , ID)
	- FluxData_y(i  ,j+1, ID);

      real_t e_new = e
	+ FluxData_x(i  ,j  , IP)
	- FluxData_x(i+1,j  , IP)
	+ FluxData_y(i  ,j  , IP)
	- FluxData_y(i  ,j+1, IP);

      real_t u_new = u
	+ FluxData_x(i  ,j  , IU)
	- FluxData_x(i+1,j  , IU)
	+ FluxData_y(i  ,j  , IU)
	- FluxData_y(i  ,j+1, IU);
      
      real_t v_new = v
	+ FluxData_x(i  ,j  , IV)
	- FluxData_x(i+1,j  , IV)
	+ FluxData_y(i  ,j  , IV)
	- FluxData_y(i  ,j+1, IV);

      // conservative variable
      HydroState UNew;
      UNew[ID]=rho;
      UNew[IP]=e;
      UNew[IU]=u;
      UNew[IV]=v;

      real_t c;
      // compute pressure from primitive variables
      HydroState QNew;
      this->computePrimitives(UNew, &c, QNew);

      // test if solution is not physically admissible (negative density or pressure)
      if (rho_new < 0 or QNew[IP] < 0)
	flag_tmp = 1.0;

      Flags(i,j,0) = flag_tmp;
      
    } // end if
    
  } // end operator () - 2d
  
  //! functor for 3d 
  template<int dim_ = dim>
  KOKKOS_INLINE_FUNCTION
  void operator()(const typename Kokkos::Impl::enable_if<dim_==3, int>::type& index) const
  {

    const int isize = this->params.isize;
    const int jsize = this->params.jsize;
    const int ksize = this->params.ksize;
    const int ghostWidth = this->params.ghostWidth;
    
    int i,j,k;
    index2coord(index,i,j,k,isize,jsize,ksize);

    // set flags to zero
    Flags(i,j,k,0) = 0.0;

    real_t flag_tmp = 0.0;
    
    if(k >= ghostWidth && k < ksize-ghostWidth  &&
       j >= ghostWidth && j < jsize-ghostWidth  &&
       i >= ghostWidth && i < isize-ghostWidth ) {

      real_t rho = Udata(i  ,j  ,k  , ID);
      real_t e   = Udata(i  ,j  ,k  , IP);
      real_t u   = Udata(i  ,j  ,k  , IU);
      real_t v   = Udata(i  ,j  ,k  , IV);
      real_t w   = Udata(i  ,j  ,k  , IW);
      
      real_t rho_new = rho
	+ FluxData_x(i  ,j  ,k  , ID)
	- FluxData_x(i+1,j  ,k  , ID)
	+ FluxData_y(i  ,j  ,k  , ID)
	- FluxData_y(i  ,j+1,k  , ID)
	+ FluxData_z(i  ,j  ,k  , ID)
	- FluxData_z(i  ,j  ,k+1, ID);

      real_t e_new = e
	+ FluxData_x(i  ,j  ,k  , IP)
	- FluxData_x(i+1,j  ,k  , IP)
	+ FluxData_y(i  ,j  ,k  , IP)
	- FluxData_y(i  ,j+1,k  , IP)
	+ FluxData_z(i  ,j  ,k  , IP)
	- FluxData_z(i  ,j  ,k+1, IP);

      real_t u_new = u
	+ FluxData_x(i  ,j  ,k  , IU)
	- FluxData_x(i+1,j  ,k  , IU)
	+ FluxData_y(i  ,j  ,k  , IU)
	- FluxData_y(i  ,j+1,k  , IU)
	+ FluxData_z(i  ,j  ,k  , IU)
	- FluxData_z(i  ,j  ,k+1, IU);
      
      real_t v_new = v
	+ FluxData_x(i  ,j  ,k  , IV)
	- FluxData_x(i+1,j  ,k  , IV)
	+ FluxData_y(i  ,j  ,k  , IV)
	- FluxData_y(i  ,j+1,k  , IV)
	+ FluxData_z(i  ,j  ,k  , IV)
	- FluxData_z(i  ,j  ,k+1, IV);

      real_t w_new = w
	+ FluxData_x(i  ,j  ,k  , IW)
	- FluxData_x(i+1,j  ,k  , IW)
	+ FluxData_y(i  ,j  ,k  , IW)
	- FluxData_y(i  ,j+1,k  , IW)
	+ FluxData_z(i  ,j  ,k  , IW)
	- FluxData_z(i  ,j  ,k+1, IW);

      // conservative variable
      HydroState UNew;
      UNew[ID]=rho;
      UNew[IP]=e;
      UNew[IU]=u;
      UNew[IV]=v;
      UNew[IW]=w;

      real_t c;
      // compute pressure from primitive variables
      HydroState QNew;
      this->computePrimitives(UNew, &c, QNew);

      // test if solution is not physically admissible, i.e.
      // negative density or pressure
      if (rho_new < 0 or QNew[IP] < 0)
	flag_tmp = 1.0;

      Flags(i,j,k,0) = flag_tmp;
      
    } // end if
    
  } // end operator () - 3d
  
  DataArray   Udata;
  DataArray   Flags;
  DataArray   FluxData_x;
  DataArray   FluxData_y;
  DataArray   FluxData_z;
  
}; // ComputeMoodFlagsUpdateFunctor

} // namespace mood

#endif // MOOD_UPDATE_FUNCTORS_H_
