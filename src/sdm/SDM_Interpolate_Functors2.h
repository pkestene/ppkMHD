#ifndef SDM_INTERPOLATE_FUNCTORS2_H_
#define SDM_INTERPOLATE_FUNCTORS2_H_

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
 * This functor takes as an input variables
 * at solution points and perform interpolation at flux points.
 *
 * It is essentially a wrapper arround interpolation method sol2flux_vector.
 *
 * Perform exactly the inverse of Interpolate_At_SolutionPoints_Functor.
 *
 */
template<int dim, int N, int dir>
class Interpolate_At_FluxPoints_Functor_v2 : public SDMBaseFunctor<dim,N> {

public:
  using typename SDMBaseFunctor<dim,N>::DataArray;
  using typename SDMBaseFunctor<dim,N>::solution_values_t;
  using typename SDMBaseFunctor<dim,N>::flux_values_t;
  
  static constexpr auto dofMapS = DofMap<dim,N>;
  static constexpr auto dofMapF = DofMapFlux<dim,N,dir>;
  
  Interpolate_At_FluxPoints_Functor(HydroParams         params,
				    SDM_Geometry<dim,N> sdm_geom,
				    DataArray           UdataSol,
				    DataArray           UdataFlux) :
    SDMBaseFunctor<dim,N>(params,sdm_geom),
    UdataSol(UdataSol),
    UdataFlux(UdataFlux)
  {};

  // static method which does it all: create and execute functor
  static void apply(HydroParams         params,
                    SDM_Geometry<dim,N> sdm_geom,
                    DataArray           UdataSol,
                    DataArray           UdataFlux,
                    int                 nbCells)
  {
    Interpolate_At_FluxPoints_Functor_v2 functor(params, sdm_geom, 
                                                 UdataSol, UdataFlux);
    Kokkos::parallel_for(nbCells, functor);
  }
  
  // =========================================================
  /*
   * 2D version.
   */
  // =========================================================
  //! functor for 2d 
  template<int dim_ = dim>
  KOKKOS_INLINE_FUNCTION
  void operator()(const typename std::enable_if<dim_==2, int>::type& index) const
  {

    const int isize = this->params.isize;
    const int jsize = this->params.jsize;

    const int nbvar = this->params.nbvar;

    // get cell index
    int ij = index / (N+1);

    // get flux point index
    int fluxId = index - (N+1)*ij;

    // get local cell coordinate
    int i,j;
    index2coord(ij,i,j,isize,jsize);

    solution_values_t sol;
    double            flux;
    
    // loop over cell DoF's
    if (dir == IX) {

      for (int idy=0; idy<N; ++idy) {

	// for each variables
	for (int ivar = 0; ivar<nbvar; ++ivar) {
	
	  // get solution values vector along X direction
	  for (int idx=0; idx<N; ++idx) {
	  
	    sol[idx] = UdataSol(i  ,j  , dofMapS(idx,idy,0,ivar));

	  }
	  
	  // interpolate at fluxId for this given variable
	  flux = this->sol2flux(sol, fluxId);

	  // positivity preserving for density
	  if (ivar==ID) {
            flux = fmax(flux, this->params.settings.smallr);
	  }
	  
	  // copy back interpolated value
          UdataFlux(i  ,j  , dofMapF(fluxId,idy,0,ivar)) = flux;
	  
	} // end for ivar
	
      } // end for idy

    } // end for dir IX

    // loop over cell DoF's
    if (dir == IY) {

      for (int idx=0; idx<N; ++idx) {

	// for each variables
	for (int ivar = 0; ivar<nbvar; ++ivar) {
	
	  // get solution values vector along Y direction
	  for (int idy=0; idy<N; ++idy) {
	  
	    sol[idy] = UdataSol(i  ,j  , dofMapS(idx,idy,0,ivar));
	    
	  }
	  
	  // interpolate at flux points for this given variable
	  flux = this->sol2flux(sol, fluxId);
	  
	  // positivity preserving for density
	  if (ivar==ID) {
            flux = fmax(flux, this->params.settings.smallr);
	  }

	  // copy back interpolated value
          UdataFlux(i  ,j  , dofMapF(idx,fluxId,0,ivar)) = flux;
	  
	} // end for ivar
	
      } // end for idx

    } // end for dir IY
    
  } // end operator () - 2d

  // =========================================================
  /*
   * 3D version.
   */
  // =========================================================
  //! functor for 3d 
  template<int dim_ = dim>
  KOKKOS_INLINE_FUNCTION
  void operator()(const typename std::enable_if<dim_==3, int>::type& index) const
  {

    const int isize = this->params.isize;
    const int jsize = this->params.jsize;
    const int ksize = this->params.ksize;

    const int nbvar = this->params.nbvar;

    // local cell index
    int i,j,k;
    index2coord(index,i,j,k,isize,jsize,ksize);

    solution_values_t sol;
    flux_values_t     flux;
    
    // loop over cell DoF's
    if (dir == IX) {

      for (int idz=0; idz<N; ++idz) {
	for (int idy=0; idy<N; ++idy) {
	  
	  // for each variables
	  for (int ivar = 0; ivar<nbvar; ++ivar) {
	    
	    // get solution values vector along X direction
	    for (int idx=0; idx<N; ++idx) {
	      
	      sol[idx] = UdataSol(i,j,k, dofMapS(idx,idy,idz,ivar));
	      
	    }
	    
	    // interpolate at flux points for this given variable
	    this->sol2flux_vector(sol, flux);
	    
	    // positivity preserving for density
	    if (ivar==ID) {
	      for (int idf=0; idf<N+1; ++idf)
		flux[idf] = fmax(flux[idf], this->params.settings.smallr);
	    }
	    
	    // copy back interpolated value
	    for (int idx=0; idx<N+1; ++idx) {
	      
	      UdataFlux(i,j,k, dofMapF(idx,idy,idz,ivar)) = flux[idx];
	      
	    }
	    
	  } // end for ivar
	  
	} // end for idy
      } // end for idz
      
    } // end for dir IX

    // loop over cell DoF's
    if (dir == IY) {

      for (int idz=0; idz<N; ++idz) {
	for (int idx=0; idx<N; ++idx) {
	  
	  // for each variables
	  for (int ivar = 0; ivar<nbvar; ++ivar) {
	    
	    // get solution values vector along Y direction
	    for (int idy=0; idy<N; ++idy) {
	      
	      sol[idy] = UdataSol(i,j,k, dofMapS(idx,idy,idz,ivar));
	    
	    }
	    
	    // interpolate at flux points for this given variable
	    this->sol2flux_vector(sol, flux);
	    
	    // positivity preserving for density
	    if (ivar==ID) {
	      for (int idf=0; idf<N+1; ++idf)
		flux[idf] = fmax(flux[idf], this->params.settings.smallr);
	    }
	    
	    // copy back interpolated value
	    for (int idy=0; idy<N+1; ++idy) {
	      
	      UdataFlux(i,j,k, dofMapF(idx,idy,idz,ivar)) = flux[idy];
	      
	    }
	  
	  } // end for ivar
	
	} // end for idx
      } // end for idz

    } // end for dir IY

    // loop over cell DoF's
    if (dir == IZ) {

      for (int idy=0; idy<N; ++idy) {
	for (int idx=0; idx<N; ++idx) {
	  
	  // for each variables
	  for (int ivar = 0; ivar<nbvar; ++ivar) {
	    
	    // get solution values vector along Y direction
	    for (int idz=0; idz<N; ++idz) {
	      
	      sol[idz] = UdataSol(i,j,k, dofMapS(idx,idy,idz,ivar));
	    
	    }
	    
	    // interpolate at flux points for this given variable
	    this->sol2flux_vector(sol, flux);
	    
	    // positivity preserving for density
	    if (ivar==ID) {
	      for (int idf=0; idf<N+1; ++idf)
		flux[idf] = fmax(flux[idf], this->params.settings.smallr);
	    }
	    
	    // copy back interpolated value
	    for (int idz=0; idz<N+1; ++idz) {
	      
	      UdataFlux(i,j,k, dofMapF(idx,idy,idz,ivar)) = flux[idz];
	      
	    }
	  
	  } // end for ivar
	
	} // end for idx
      } // end for idz

    } // end for dir IZ

  } // end operator () - 3d
  
  DataArray UdataSol, UdataFlux;

}; // class Interpolate_At_FluxPoints_Functor

enum Interpolation_type_t {
  INTERPOLATE_DERIVATIVE=0,
  INTERPOLATE_SOLUTION=1,
  INTERPOLATE_DERIVATIVE_NEGATIVE=2,
  INTERPOLATE_SOLUTION_NEGATIVE=3
};

/*************************************************/
/*************************************************/
/*************************************************/
/**
 * This functor takes as an input variables
 * at flux points and perform interpolation at solution points, and
 * accumulates result in output array (UdataSol).
 *
 * Its is essentially a wrapper arround interpolation method flux2sol_vector.
 *
 * Perform exactly the inverse of Interpolate_At_FluxPoints_Functor
 */
template<int dim, int N, int dir,
	 Interpolation_type_t itype=INTERPOLATE_DERIVATIVE>
class Interpolate_At_SolutionPoints_Functor_v2 : public SDMBaseFunctor<dim,N> {

public:
  using typename SDMBaseFunctor<dim,N>::DataArray;
  using typename SDMBaseFunctor<dim,N>::solution_values_t;
  using typename SDMBaseFunctor<dim,N>::flux_values_t;
  
  static constexpr auto dofMapS = DofMap<dim,N>;
  static constexpr auto dofMapF = DofMapFlux<dim,N,dir>;
  
  Interpolate_At_SolutionPoints_Functor(HydroParams         params,
					SDM_Geometry<dim,N> sdm_geom,
					DataArray           UdataFlux,
					DataArray           UdataSol) :
    SDMBaseFunctor<dim,N>(params,sdm_geom),
    UdataFlux(UdataFlux),
    UdataSol(UdataSol)
  {};

  // static method which does it all: create and execute functor
  static void apply(HydroParams         params,
                    SDM_Geometry<dim,N> sdm_geom,
                    DataArray           UdataFlux,
                    DataArray           UdataSol,
                    int                 nbCells)
  {
    Interpolate_At_SolutionPoints_Functor_v2 functor(params, sdm_geom, 
                                                     UdataFlux, UdataSol);
    Kokkos::parallel_for(nbCells, functor);
  }
  
  // =========================================================
  /*
   * 2D version.
   */
  // =========================================================
  //! functor for 2d 
  template<int dim_ = dim>
  KOKKOS_INLINE_FUNCTION
  void operator()(const typename std::enable_if<dim_==2, int>::type& index) const
  {

    const int isize = this->params.isize;
    const int jsize = this->params.jsize;

    const int nbvar = this->params.nbvar;

    // rescale factor for derivative
    real_t rescale = 1.0/this->params.dx;
    if (dir == IY)
      rescale = 1.0/this->params.dy;
    
    // local cell index
    int i,j;
    index2coord(index,i,j,isize,jsize);

    solution_values_t sol;
    flux_values_t     flux;
    
    // loop over cell DoF's
    if (dir == IX) {

      for (int idy=0; idy<N; ++idy) {

	// for each variables
	for (int ivar = 0; ivar<nbvar; ++ivar) {
	
	  // get values at flux point along X direction
	  for (int idx=0; idx<N+1; ++idx) {
	  
	    flux[idx] = UdataFlux(i  ,j  , dofMapF(idx,idy,0,ivar));
	    
	  }
	  
	  // interpolate at flux points for this given variable
	  if (itype==INTERPOLATE_SOLUTION or
	      itype==INTERPOLATE_SOLUTION_NEGATIVE)
	    this->flux2sol_vector(flux, sol);
	  else
	    this->flux2sol_derivative_vector(flux,sol,rescale);
	  
	  // copy back interpolated value
	  for (int idx=0; idx<N; ++idx) {
	    
	    if (itype==INTERPOLATE_DERIVATIVE_NEGATIVE or
		itype==INTERPOLATE_SOLUTION_NEGATIVE)
	      UdataSol(i  ,j  , dofMapS(idx,idy,0,ivar)) -= sol[idx];
	    else
	      UdataSol(i  ,j  , dofMapS(idx,idy,0,ivar)) += sol[idx];
	    
	  }
	  
	} // end for ivar
	
      } // end for idy

    } // end for dir IX

    // loop over cell DoF's
    if (dir == IY) {

      for (int idx=0; idx<N; ++idx) {

	// for each variables
	for (int ivar = 0; ivar<nbvar; ++ivar) {
	
	  // get values at flux point along Y direction
	  for (int idy=0; idy<N+1; ++idy) {
	  
	    flux[idy] = UdataFlux(i  ,j  , dofMapF(idx,idy,0,ivar));
	    
	  }
	  
	  // interpolate at flux points for this given variable
	  if (itype==INTERPOLATE_SOLUTION or
	      itype==INTERPOLATE_SOLUTION_NEGATIVE)
	    this->flux2sol_vector(flux, sol);
	  else
	    this->flux2sol_derivative_vector(flux,sol,rescale);
	  
	  // copy back interpolated value
	  for (int idy=0; idy<N; ++idy) {
	    
	    if (itype==INTERPOLATE_DERIVATIVE_NEGATIVE or
		itype==INTERPOLATE_SOLUTION_NEGATIVE)
	      UdataSol(i  ,j  , dofMapS(idx,idy,0,ivar)) -= sol[idy];
	    else
	      UdataSol(i  ,j  , dofMapS(idx,idy,0,ivar)) += sol[idy];
	    
	  }
	  
	} // end for ivar
	
      } // end for idx

    } // end for dir IY
    
  } // end operator () - 2d

  // =========================================================
  /*
   * 3D version.
   */
  // =========================================================
  //! functor for 3d 
  template<int dim_ = dim>
  KOKKOS_INLINE_FUNCTION
  void operator()(const typename std::enable_if<dim_==3, int>::type& index) const
  {

    const int isize = this->params.isize;
    const int jsize = this->params.jsize;
    const int ksize = this->params.ksize;

    const int nbvar = this->params.nbvar;

    // rescale factor for derivative
    real_t rescale = 1.0/this->params.dx;
    if (dir == IY)
      rescale = 1.0/this->params.dy;
    if (dir == IZ)
      rescale = 1.0/this->params.dz;

    // local cell index
    int i,j,k;
    index2coord(index,i,j,k,isize,jsize,ksize);

    solution_values_t sol;
    flux_values_t     flux;
    
    // loop over cell DoF's
    if (dir == IX) {

      for (int idz=0; idz<N; ++idz) {
	for (int idy=0; idy<N; ++idy) {
	  
	  // for each variables
	  for (int ivar = 0; ivar<nbvar; ++ivar) {
	    
	    // get values at flux point along X direction
	    for (int idx=0; idx<N+1; ++idx) {
	      
	      flux[idx] = UdataFlux(i,j,k, dofMapF(idx,idy,idz,ivar));
	    
	    }
	  
	    // interpolate at flux points for this given variable
	    if (itype==INTERPOLATE_SOLUTION or
		itype==INTERPOLATE_SOLUTION_NEGATIVE)
	      this->flux2sol_vector(flux, sol);
	    else
	      this->flux2sol_derivative_vector(flux,sol,rescale);
	    
	    // copy back interpolated value
	    for (int idx=0; idx<N; ++idx) {
	    
	    if (itype==INTERPOLATE_DERIVATIVE_NEGATIVE or
		itype==INTERPOLATE_SOLUTION_NEGATIVE)
	      UdataSol(i,j,k, dofMapS(idx,idy,idz,ivar)) -= sol[idx];
	    else
	      UdataSol(i,j,k, dofMapS(idx,idy,idz,ivar)) += sol[idx];
	    
	    }
	    
	  } // end for ivar
	  
	} // end for idy
      } // end for idz

    } // end for dir IX

    // loop over cell DoF's
    if (dir == IY) {

      for (int idz=0; idz<N; ++idz) {
	for (int idx=0; idx<N; ++idx) {

	  // for each variables
	  for (int ivar = 0; ivar<nbvar; ++ivar) {
	    
	    // get values at flux point along Y direction
	    for (int idy=0; idy<N+1; ++idy) {
	      
	      flux[idy] = UdataFlux(i,j,k, dofMapF(idx,idy,idz,ivar));
	      
	    }
	    
	    // interpolate at flux points for this given variable
	    if (itype==INTERPOLATE_SOLUTION or
		itype==INTERPOLATE_SOLUTION_NEGATIVE)
	      this->flux2sol_vector(flux, sol);
	    else
	      this->flux2sol_derivative_vector(flux,sol,rescale);
	    
	    // copy back interpolated value
	    for (int idy=0; idy<N; ++idy) {
	    
	    if (itype==INTERPOLATE_DERIVATIVE_NEGATIVE or
		itype==INTERPOLATE_SOLUTION_NEGATIVE)
	      UdataSol(i,j,k, dofMapS(idx,idy,idz,ivar)) -= sol[idy];
	    else
	      UdataSol(i,j,k, dofMapS(idx,idy,idz,ivar)) += sol[idy];

	    }
	    
	  } // end for ivar
	  
	} // end for idx
      } // end for idz

    } // end for dir IY

    // loop over cell DoF's
    if (dir == IZ) {

      for (int idy=0; idy<N; ++idy) {
	for (int idx=0; idx<N; ++idx) {

	  // for each variables
	  for (int ivar = 0; ivar<nbvar; ++ivar) {
	    
	    // get values at flux point along Y direction
	    for (int idz=0; idz<N+1; ++idz) {
	      
	      flux[idz] = UdataFlux(i,j,k, dofMapF(idx,idy,idz,ivar));
	      
	    }
	    
	    // interpolate at flux points for this given variable
	    if (itype==INTERPOLATE_SOLUTION or
		itype==INTERPOLATE_SOLUTION_NEGATIVE)
	      this->flux2sol_vector(flux, sol);
	    else
	      this->flux2sol_derivative_vector(flux,sol,rescale);
	    
	    // copy back interpolated value
	    for (int idz=0; idz<N; ++idz) {
	    
	    if (itype==INTERPOLATE_DERIVATIVE_NEGATIVE or
		itype==INTERPOLATE_SOLUTION_NEGATIVE)
	      UdataSol(i,j,k, dofMapS(idx,idy,idz,ivar)) -= sol[idz];
	    else
	      UdataSol(i,j,k, dofMapS(idx,idy,idz,ivar)) += sol[idz];
	    
	    }
	    
	  } // end for ivar
	  
	} // end for idx
      } // end for idy

    } // end for dir IZ

  } // end operator () - 3d
  
  DataArray UdataFlux, UdataSol;

}; // Interpolate_At_SolutionPoints_Functor

} // namespace sdm

#endif // SDM_INTERPOLATE_FUNCTORS2_H_
