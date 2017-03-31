#ifndef MOOD_UPDATE_FUNCTORS_H_
#define MOOD_UPDATE_FUNCTORS_H_

#include "shared/kokkos_shared.h"
#include "shared/HydroParams.h"
#include "shared/HydroState.h"

namespace mood {

// =======================================================================
// =======================================================================
class UpdateFunctor2D
{

public:

  UpdateFunctor2D(HydroParams params,
		  DataArray2d Udata,
		  DataArray2d FluxData_x,
		  DataArray2d FluxData_y) :
    params(params),
    Udata(Udata),
    FluxData_x(FluxData_x),
    FluxData_y(FluxData_y)
  {};
  
  KOKKOS_INLINE_FUNCTION
  void operator()(const int& index) const
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
  
  HydroParams params;
  DataArray2d Udata;
  DataArray2d FluxData_x;
  DataArray2d FluxData_y;
  
}; // UpdateFunctor2D

// =======================================================================
// =======================================================================
/**
 * This functor tries to perform update on density, if density or internal energy
 * becomes negative, we flag the cells for recompute.
 */
class ComputeMoodFlagsUpdateFunctor2D
{

public:

  ComputeMoodFlagsUpdateFunctor2D(HydroParams params,
				  DataArray2d Udata,
				  DataArray2d Flags,
				  DataArray2d FluxData_x,
				  DataArray2d FluxData_y) :
    params(params),
    Udata(Udata),
    Flags(Flags),
    FluxData_x(FluxData_x),
    FluxData_y(FluxData_y)
  {};
  
  KOKKOS_INLINE_FUNCTION
  void operator()(const int& index) const
  {
    const int isize = params.isize;
    const int jsize = params.jsize;
    const int ghostWidth = params.ghostWidth;
    
    int i,j;
    index2coord(index,i,j,isize,jsize);

    // set flags to zero
    Flags(i,j,0) = 0.0;

    real_t flag_tmp = 0.0;
    
    if(j >= ghostWidth && j < jsize-ghostWidth  &&
       i >= ghostWidth && i < isize-ghostWidth ) {

      real_t rho = Udata(i  ,j  , ID);
      real_t e   = Udata(i  ,j  , IP);

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

      if (rho_new < 0 or e_new < 0)
	flag_tmp = 1.0;

      Flags(i,j,0) = flag_tmp;
      
    } // end if
    
  } // end operator ()
  
  HydroParams params;
  DataArray2d Udata;
  DataArray2d Flags;
  DataArray2d FluxData_x;
  DataArray2d FluxData_y;
  
}; // ComputeMoodFlagsUpdateFunctor2D

} // namespace mood

#endif // MOOD_UPDATE_FUNCTORS_H_
