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
		  DataArray2d FluxData_y,
		  real_t dtdx,
		  real_t dtdy) :
    params(params),
    Udata(Udata),
    FluxData_x(FluxData_x),
    FluxData_y(FluxData_y),
    dtdx(dtdx),
    dtdy(dtdy)
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

      Udata(i  ,j  , ID) +=  FluxData_x(i  ,j  , ID) * dtdx;
      Udata(i  ,j  , IP) +=  FluxData_x(i  ,j  , IP) * dtdx;
      Udata(i  ,j  , IU) +=  FluxData_x(i  ,j  , IU) * dtdx;
      Udata(i  ,j  , IV) +=  FluxData_x(i  ,j  , IV) * dtdx;

      Udata(i  ,j  , ID) -=  FluxData_x(i+1,j  , ID) * dtdx;
      Udata(i  ,j  , IP) -=  FluxData_x(i+1,j  , IP) * dtdx;
      Udata(i  ,j  , IU) -=  FluxData_x(i+1,j  , IU) * dtdx;
      Udata(i  ,j  , IV) -=  FluxData_x(i+1,j  , IV) * dtdx;
      
      Udata(i  ,j  , ID) +=  FluxData_y(i  ,j  , ID) * dtdy;
      Udata(i  ,j  , IP) +=  FluxData_y(i  ,j  , IP) * dtdy;
      Udata(i  ,j  , IU) +=  FluxData_y(i  ,j  , IV) * dtdy; //
      Udata(i  ,j  , IV) +=  FluxData_y(i  ,j  , IU) * dtdy; //
      
      Udata(i  ,j  , ID) -=  FluxData_y(i  ,j+1, ID) * dtdy;
      Udata(i  ,j  , IP) -=  FluxData_y(i  ,j+1, IP) * dtdy;
      Udata(i  ,j  , IU) -=  FluxData_y(i  ,j+1, IV) * dtdy; //
      Udata(i  ,j  , IV) -=  FluxData_y(i  ,j+1, IU) * dtdy; //

    } // end if
    
  } // end operator ()
  
  HydroParams params;
  DataArray2d Udata;
  DataArray2d FluxData_x;
  DataArray2d FluxData_y;
  real_t dtdx, dtdy;
  
}; // UpdateFunctor2D

// =======================================================================
// =======================================================================
/**
 * This functor tries to perform update on density, if density becomes negative, 
 * we flag the cells for recompute.
 */
class ComputeMoodFlagsUpdateFunctor2D
{

public:

  ComputeMoodFlagsUpdateFunctor2D(HydroParams params,
				  DataArray2d Udata,
				  DataArray2d Flags,
				  DataArray2d FluxData_x,
				  DataArray2d FluxData_y,
				  real_t dtdx,
				  real_t dtdy) :
    params(params),
    Udata(Udata),
    Flags(Flags),
    FluxData_x(FluxData_x),
    FluxData_y(FluxData_y),
    dtdx(dtdx),
    dtdy(dtdy)
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

      real_t rho_new = rho
	+ FluxData_x(i  ,j  , ID) * dtdx
	- FluxData_x(i+1,j  , ID) * dtdx
	+ FluxData_y(i  ,j  , ID) * dtdy
	- FluxData_y(i  ,j+1, ID) * dtdy; 

      if (rho_new < 0)
	flag_tmp = 1.0;

      Flags(i,j,0) = flag_tmp;
      
    } // end if
    
  } // end operator ()
  
  HydroParams params;
  DataArray2d Udata;
  DataArray2d Flags;
  DataArray2d FluxData_x;
  DataArray2d FluxData_y;
  real_t      dtdx, dtdy;
  
}; // ComputeMoodFlagsUpdateFunctor2D

} // namespace mood

#endif // MOOD_UPDATE_FUNCTORS_H_
