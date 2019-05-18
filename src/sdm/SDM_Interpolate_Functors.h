#ifndef SDM_INTERPOLATE_FUNCTORS_H_
#define SDM_INTERPOLATE_FUNCTORS_H_

#include <limits> // for std::numeric_limits
#ifdef __CUDA_ARCH__
#include <math_constants.h> // for cuda math constants, e.g. CUDART_INF
#endif // __CUDA_ARCH__

#include "shared/kokkos_shared.h"
#include "sdm/SDMBaseFunctor.h"

#include "sdm/SDM_Geometry.h"
#include "sdm/sdm_shared.h"

#include "shared/EulerEquations.h"

namespace sdm {

/*************************************************/
/*************************************************/
/*************************************************/
/**
 * This functor takes as an input variables
 * at solution points (UdataSol)and perform interpolation at 
 * flux points (UdataFlux).
 * It used Kokkos range execution policy, with a number of iterations
 * mapping outputs, i.e. the flux Dof locations.
 * 
 * It is essentially a wrapper arround interpolation method sol2flux_vector.
 *
 * Perform exactly the inverse of Interpolate_At_SolutionPoints_Functor.
 *
 */
template<int dim, int N, int dir>
class Interpolate_At_FluxPoints_Functor : public SDMBaseFunctor<dim,N> {

public:
  using typename SDMBaseFunctor<dim,N>::solution_values_t;
  using typename SDMBaseFunctor<dim,N>::flux_values_t;  
  
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
                    DataArray           UdataFlux)
  {
    int nbDofsPerCellFlux = (dim==2) ? N*(N+1) : N*N*(N+1);
    int nbDofsFlux = dim==2 ? 
      nbDofsPerCellFlux*params.isize*params.jsize : 
      nbDofsPerCellFlux*params.isize*params.jsize*params.ksize;

    Interpolate_At_FluxPoints_Functor functor(params, sdm_geom, 
                                              UdataSol, UdataFlux);
    Kokkos::parallel_for("Interpolate_At_FluxPoints_Functor", nbDofsFlux, functor);
  }
  
  // =========================================================
  /*
   * 2D version.
   */
  // =========================================================
  //! functor for 2d 
  template<int dim_ = dim>
  KOKKOS_INLINE_FUNCTION
  void operator()(const typename Kokkos::Impl::enable_if<dim_==2, int>::type& index) const
  {

    const int isize = this->params.isize;
    //const int jsize = this->params.jsize;

    const int nbvar = this->params.nbvar;

    int iDof, iCell;
    index_to_iDof_iCell(index,N*(N+1),iDof,iCell);

    // cell coord
    int i,j;
    iCell_to_coord(iCell,isize,i,j);

    // Dof coord
    int idx,idy;
    iDof_to_coord_flux<dir>(iDof,N,idx,idy);

    // input
    solution_values_t sol;

    // output interpolated value
    real_t            flux;
    
    // loop over cell DoF's
    if (dir == IX) {
      
      // for each variables
      for (int ivar = 0; ivar<nbvar; ++ivar) {
        
        // get solution values vector along X direction
        for (int idf=0; idf<N; ++idf) {
          
          sol[idf] = UdataSol(idf+N*idy,  iCell, ivar);
          
        }
	
        // interpolate at flux points for this given variable
        flux = this->sol2flux(sol, idx);
        
        // positivity preserving for density
        if (ivar==ID) {
          flux = fmax(flux, this->params.settings.smallr);
        }
        
        // copy back interpolated value
        UdataFlux(iDof, iCell, ivar) = flux;
	  
      } // end for ivar
	
    } // end for dir IX

    // loop over cell DoF's
    if (dir == IY) {
      
      // for each variables
      for (int ivar = 0; ivar<nbvar; ++ivar) {
        
        // get solution values vector along Y direction
        for (int idf=0; idf<N; ++idf) {
          
          sol[idf] = UdataSol(idx+N*idf, iCell, ivar);
	  
        }
	
        // interpolate at flux points for this given variable
        flux = this->sol2flux(sol, idy);
	
        // positivity preserving for density
        if (ivar==ID) {
          flux = fmax(flux, this->params.settings.smallr);
        }
        
        // copy back interpolated value
        UdataFlux(iDof, iCell, ivar) = flux;
          
      } // end for ivar
	
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
  void operator()(const typename Kokkos::Impl::enable_if<dim_==3, int>::type& index) const
  {

    const int isize = this->params.isize;
    const int jsize = this->params.jsize;
    //const int ksize = this->params.ksize;
    
    const int nbvar = this->params.nbvar;

    int iDof, iCell;
    index_to_iDof_iCell(index,N*N*(N+1),iDof,iCell);

    // cell coord
    int i,j,k;
    iCell_to_coord(iCell,isize,jsize,i,j,k);

    // Dof coord
    int idx,idy,idz;
    iDof_to_coord_flux<dir>(iDof,N,idx,idy,idz);

    // input
    solution_values_t sol;
    
    // output interpolated value
    real_t            flux;
    
    // loop over cell DoF's
    if (dir == IX) {

      // for each variables
      for (int ivar = 0; ivar<nbvar; ++ivar) {
        
        // get solution values vector along X direction
        for (int idf=0; idf<N; ++idf) {
          
          sol[idf] = UdataSol(idf+N*idy+N*N*idz, iCell, ivar);
	  
        }
	
        // interpolate at flux points for this given variable
        flux = this->sol2flux(sol, idx);
	
        // positivity preserving for density
        if (ivar==ID) {
          flux = fmax(flux, this->params.settings.smallr);
        }
	
        // copy back interpolated value
        for (int idx=0; idx<N+1; ++idx) {
          
          UdataFlux(iDof, iCell, ivar) = flux;
	  
        }
	
      } // end for ivar
            
    } // end for dir IX

    // loop over cell DoF's
    if (dir == IY) {

      // for each variables
      for (int ivar = 0; ivar<nbvar; ++ivar) {
        
        // get solution values vector along Y direction
        for (int idf=0; idf<N; ++idf) {
          
          sol[idf] = UdataSol(idx+N*idf+N*N*idz, iCell, ivar);
	  
        }
	
        // interpolate at flux points for this given variable
        flux = this->sol2flux(sol, idy);
	
        // positivity preserving for density
        if (ivar==ID) {
          flux = fmax(flux, this->params.settings.smallr);
        }
	
        // copy back interpolated value
        UdataFlux(iDof, iCell, ivar) = flux;
	
      } // end for ivar
      
    } // end for dir IY
    
    // loop over cell DoF's
    if (dir == IZ) {
      
      // for each variables
      for (int ivar = 0; ivar<nbvar; ++ivar) {
        
        // get solution values vector along Y direction
        for (int idf=0; idf<N; ++idf) {
          
          sol[idf] = UdataSol(idx+N*idy+N*N*idf, iCell, ivar);
	  
        }
	
        // interpolate at flux points for this given variable
        flux = this->sol2flux(sol, idz);
	
        // positivity preserving for density
        if (ivar==ID) {
          flux = fmax(flux, this->params.settings.smallr);
        }
	
        // copy back interpolated value
        UdataFlux(iDof, iCell, ivar) = flux;
	
      } // end for ivar
      
    } // end for dir IZ
    
  } // end operator () - 3d
  
  DataArray UdataSol, UdataFlux;

}; // class Interpolate_At_FluxPoints_Functor

enum Interpolation_type_t {
  INTERPOLATE_DERIVATIVE=0,
  INTERPOLATE_SOLUTION=1,
  INTERPOLATE_DERIVATIVE_NEGATIVE=2,
  INTERPOLATE_SOLUTION_NEGATIVE=3,
  INTERPOLATE_SOLUTION_REGULAR=4
};

/*************************************************/
/*************************************************/
/*************************************************/
/**
 * This functor takes as an input variables
 * at flux points (UdataFlux) and perform interpolation at solution
 * points, and accumulates result in output array (UdataSol).
 * It used Kokkos range execution policy, with a number of iterations
 * mapping outputs, i.e. the solution Dof locations.
  *
 * Its is essentially a wrapper arround interpolation method flux2sol_vector.
 *
 * Perform exactly the inverse of Interpolate_At_FluxPoints_Functor
 */
template<int dim, int N, int dir,
	 Interpolation_type_t itype=INTERPOLATE_DERIVATIVE>
class Interpolate_At_SolutionPoints_Functor : public SDMBaseFunctor<dim,N> {

public:
  using typename SDMBaseFunctor<dim,N>::solution_values_t;
  using typename SDMBaseFunctor<dim,N>::flux_values_t;
  
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
                    DataArray           UdataSol)
  {
    int nbDofsPerCell = (dim==2) ? N*N : N*N*N;
    int nbDofs = dim==2 ? 
      nbDofsPerCell*params.isize*params.jsize : 
      nbDofsPerCell*params.isize*params.jsize*params.ksize;
    
    Interpolate_At_SolutionPoints_Functor functor(params, sdm_geom,
                                                  UdataFlux, UdataSol);
    Kokkos::parallel_for("Interpolate_At_SolutionPoints_Functor",nbDofs, functor);
  }
  
  // =========================================================
  /*
   * 2D version.
   */
  // =========================================================
  //! functor for 2d 
  template<int dim_ = dim>
  KOKKOS_INLINE_FUNCTION
  void operator()(const typename Kokkos::Impl::enable_if<dim_==2, int>::type& index) const
  {

    const int isize = this->params.isize;
    //const int jsize = this->params.jsize;

    const int nbvar = this->params.nbvar;

    // rescale factor for derivative
    real_t rescale = 1.0/this->params.dx;
    if (dir == IY)
      rescale = 1.0/this->params.dy;
    
    int iDof, iCell;
    index_to_iDof_iCell(index,N*N,iDof,iCell);

    // cell coord
    int i,j;
    iCell_to_coord(iCell,isize,i,j);

    // Dof coord
    int idx,idy;
    iDof_to_coord(iDof,N,idx,idy);

    // ouptut
    real_t sol;
    
    // input
    flux_values_t flux;
    
    // loop over cell DoF's
    if (dir == IX) {
      
      // for each variables
      for (int ivar = 0; ivar<nbvar; ++ivar) {
        
        // get values at flux point along X direction
        for (int id=0; id<N+1; ++id) {
          
          flux[id] = UdataFlux(id+(N+1)*idy, iCell, ivar);
	  
        }
	
        // interpolate at flux points for this given variable
        if (itype==INTERPOLATE_SOLUTION or
            itype==INTERPOLATE_SOLUTION_NEGATIVE or
            itype==INTERPOLATE_SOLUTION_REGULAR)
          sol = this->flux2sol(flux, idx);
        else
          sol = this->flux2sol_derivative(flux,idx,rescale);
        
        // copy back interpolated value
        if (itype==INTERPOLATE_DERIVATIVE_NEGATIVE or
            itype==INTERPOLATE_SOLUTION_NEGATIVE)
          UdataSol(iDof, iCell, ivar) -= sol;
        else if (itype==INTERPOLATE_SOLUTION_REGULAR)
          UdataSol(iDof, iCell, ivar) = sol;
        else
          UdataSol(iDof, iCell, ivar) += sol;
	
      } // end for ivar
	
    } // end for dir IX

    // loop over cell DoF's
    if (dir == IY) {

      // for each variables
      for (int ivar = 0; ivar<nbvar; ++ivar) {
        
        // get values at flux point along Y direction
        for (int id=0; id<N+1; ++id) {
          
          flux[id] = UdataFlux(idx+N*id, iCell, ivar);
	  
        }
	
        // interpolate at flux points for this given variable
        if (itype==INTERPOLATE_SOLUTION or
            itype==INTERPOLATE_SOLUTION_NEGATIVE or
            itype==INTERPOLATE_SOLUTION_REGULAR)
          sol = this->flux2sol(flux, idy);
        else
          sol = this->flux2sol_derivative(flux,idy,rescale);
        
        // copy back interpolated value
        if (itype==INTERPOLATE_DERIVATIVE_NEGATIVE or
            itype==INTERPOLATE_SOLUTION_NEGATIVE)
          UdataSol(iDof, iCell, ivar) -= sol;
        else if (itype==INTERPOLATE_SOLUTION_REGULAR)
          UdataSol(iDof, iCell, ivar) = sol;
        else
          UdataSol(iDof, iCell, ivar) += sol;
	
      } // end for ivar
	
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
  void operator()(const typename Kokkos::Impl::enable_if<dim_==3, int>::type& index) const
  {

    const int isize = this->params.isize;
    const int jsize = this->params.jsize;
    //const int ksize = this->params.ksize;

    const int nbvar = this->params.nbvar;

    // rescale factor for derivative
    real_t rescale = 1.0/this->params.dx;
    if (dir == IY)
      rescale = 1.0/this->params.dy;
    if (dir == IZ)
      rescale = 1.0/this->params.dz;

    int iDof, iCell;
    index_to_iDof_iCell(index,N*N*N,iDof,iCell);

    // cell coord
    int i,j,k;
    iCell_to_coord(iCell,isize,jsize,i,j,k);

    // Dof coord
    int idx,idy,idz;
    iDof_to_coord(iDof,N,idx,idy,idz);

    // ouptut
    real_t sol;

    // input
    flux_values_t flux;
    
    // loop over cell DoF's
    if (dir == IX) {
      
      // for each variables
      for (int ivar = 0; ivar<nbvar; ++ivar) {
        
        // get values at flux point along X direction
        for (int id=0; id<N+1; ++id) {
          
          flux[id] = UdataFlux(id+(N+1)*idy+N*(N+1)*idz, iCell, ivar);
	  
        }
	
        // interpolate at flux points for this given variable
        if (itype==INTERPOLATE_SOLUTION or
            itype==INTERPOLATE_SOLUTION_NEGATIVE or
            itype==INTERPOLATE_SOLUTION_REGULAR)
          sol = this->flux2sol(flux, idx);
        else
          sol = this->flux2sol_derivative(flux,idx,rescale);
        
        // copy back interpolated value
        if (itype==INTERPOLATE_DERIVATIVE_NEGATIVE or
            itype==INTERPOLATE_SOLUTION_NEGATIVE)
          UdataSol(iDof, iCell, ivar) -= sol;
        else if (itype==INTERPOLATE_SOLUTION_REGULAR)
          UdataSol(iDof, iCell, ivar) = sol;
        else
          UdataSol(iDof, iCell, ivar) += sol;
	
      } // end for ivar	  

    } // end for dir IX

    // loop over cell DoF's
    if (dir == IY) {

      // for each variables
      for (int ivar = 0; ivar<nbvar; ++ivar) {
        
        // get values at flux point along Y direction
        for (int id=0; id<N+1; ++id) {
          
          flux[id] = UdataFlux(idx+N*id+N*(N+1)*idz, iCell, ivar);
	  
        }
	
        // interpolate at flux points for this given variable
        if (itype==INTERPOLATE_SOLUTION or
            itype==INTERPOLATE_SOLUTION_NEGATIVE or
            itype==INTERPOLATE_SOLUTION_REGULAR)
          sol = this->flux2sol(flux, idy);
        else
          sol = this->flux2sol_derivative(flux,idy,rescale);
        
        // copy back interpolated value
        if (itype==INTERPOLATE_DERIVATIVE_NEGATIVE or
            itype==INTERPOLATE_SOLUTION_NEGATIVE)
          UdataSol(iDof,iCell, ivar) -= sol;
        else if (itype==INTERPOLATE_SOLUTION_REGULAR)
          UdataSol(iDof,iCell, ivar) = sol;
        else
          UdataSol(iDof,iCell, ivar) += sol;
	
      } // end for ivar
      
    } // end for dir IY
    
    // loop over cell DoF's
    if (dir == IZ) {
      
      // for each variables
      for (int ivar = 0; ivar<nbvar; ++ivar) {
        
        // get values at flux point along Y direction
        for (int id=0; id<N+1; ++id) {
          
          flux[id] = UdataFlux(idx+N*idy+N*N*id, iCell, ivar);
	  
        }
	
        // interpolate at flux points for this given variable
        if (itype==INTERPOLATE_SOLUTION or
            itype==INTERPOLATE_SOLUTION_NEGATIVE or
            itype==INTERPOLATE_SOLUTION_REGULAR)
          sol = this->flux2sol(flux, idz);
        else
          sol = this->flux2sol_derivative(flux,idz,rescale);
        
        // copy back interpolated value
        if (itype==INTERPOLATE_DERIVATIVE_NEGATIVE or
            itype==INTERPOLATE_SOLUTION_NEGATIVE)
          UdataSol(iDof,iCell, ivar) -= sol;
        else if (itype==INTERPOLATE_SOLUTION_REGULAR)
          UdataSol(iDof,iCell, ivar) = sol;
        else
          UdataSol(iDof,iCell, ivar) += sol;
	
      } // end for ivar

    } // end for dir IZ

  } // end operator () - 3d
  
  DataArray UdataFlux, UdataSol;

}; // Interpolate_At_SolutionPoints_Functor

} // namespace sdm

#endif // SDM_INTERPOLATE_FUNCTORS_H_
