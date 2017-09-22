#ifndef SDM_INTERPOLATE_VISCOUS_FUNCTORS_H_
#define SDM_INTERPOLATE_VISCOUS_FUNCTORS_H_

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
 * This functor takes as an input conservative variables
 * at solution points and perform interpolation of velocities
 * at flux points. What happends at cell borders is the subject
 * of an another functor : Average_component_at_cell_borders_functor
 *
 * It is essentially a wrapper arround interpolation method sol2flux_vector.
 *
 * Please note that velocity components in the flux out array must be addressed through 
 * IGU, IGV, IGW defined in enum class VarIndexGrad2d and VarIndexGrad3d.
 *
 * \sa Interpolate_At_FluxPoints_Functor
 *
 */
template<int dim, int N, int dir>
class Interpolate_velocities_Sol2Flux_Functor : public SDMBaseFunctor<dim,N> {

public:
  using typename SDMBaseFunctor<dim,N>::DataArray;
  using typename SDMBaseFunctor<dim,N>::solution_values_t;
  using typename SDMBaseFunctor<dim,N>::flux_values_t;
  
  using SDMBaseFunctor<dim,N>::IGU;
  using SDMBaseFunctor<dim,N>::IGV;
  using SDMBaseFunctor<dim,N>::IGW;
  using SDMBaseFunctor<dim,N>::IGT;

  static constexpr auto dofMapS = DofMap<dim,N>;
  static constexpr auto dofMapF = DofMapFlux<dim,N,dir>;

  /**
   * \param[in] UdataSol array of conservative variables at solution points.
   * \param[out] UdataFlux array of velocity components at flux points.
   *
   * Please note that velocity components in the flux out array must be addressed through 
   * IGU, IGV, IGW defined in enum class VarIndexGrad2d and VarIndexGrad3d.
   *
   * This means UdataFlux should have been allocated with a number of fields of at leat "dim".
   */
  Interpolate_velocities_Sol2Flux_Functor(HydroParams         params,
					  SDM_Geometry<dim,N> sdm_geom,
					  DataArray           UdataSol,
					  DataArray           UdataFlux) :
    SDMBaseFunctor<dim,N>(params,sdm_geom),
    UdataSol(UdataSol),
    UdataFlux(UdataFlux)
  {};
  
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
    const int jsize = this->params.jsize;

    // local cell index
    int i,j;
    index2coord(index,i,j,isize,jsize);

    solution_values_t sol;
    flux_values_t     flux;

    const Kokkos::Array<int,2> ivar_in_list  = {IU,  IV};
    const Kokkos::Array<int,2> ivar_out_list = {IGU, IGV};
    
    // loop over cell DoF's
    if (dir == IX) {

      for (int idy=0; idy<N; ++idy) {

	// for each velocity components
	for (int icomp = 0; icomp<dim; ++icomp) {

	  const int ivar_in  = ivar_in_list[icomp];
	  const int ivar_out = ivar_out_list[icomp];
	  
	  // get solution values vector along X direction
	  for (int idx=0; idx<N; ++idx) {

	    // divide momentum by density to obtain velocity
	    sol[idx] =
	      UdataSol(i  ,j  , dofMapS(idx,idy,0,ivar_in)) /
	      UdataSol(i  ,j  , dofMapS(idx,idy,0,ID));

	  }
	  
	  // interpolate at flux points for this given variable
	  this->sol2flux_vector(sol, flux);
	  
	  // copy back interpolated value
	  for (int idx=0; idx<N+1; ++idx) {
	    
	    UdataFlux(i  ,j  , dofMapF(idx,idy,0,ivar_out)) = flux[idx];
	    
	  } // end for idx
	  
	} // end for icomp
	
      } // end for idy

    } // end for dir IX

    // loop over cell DoF's
    if (dir == IY) {

      for (int idx=0; idx<N; ++idx) {

	// for each variables
	for (int icomp = 0; icomp<dim; ++icomp) {

	  const int ivar_in  = ivar_in_list[icomp];
	  const int ivar_out = ivar_out_list[icomp];
	
	  // get solution values vector along Y direction
	  for (int idy=0; idy<N; ++idy) {
	  
	    sol[idy] =
	      UdataSol(i  ,j  , dofMapS(idx,idy,0,ivar_in)) /
	      UdataSol(i  ,j  , dofMapS(idx,idy,0,ID));
	    
	  }
	  
	  // interpolate at flux points for this given variable
	  this->sol2flux_vector(sol, flux);
	  
	  // copy back interpolated value
	  for (int idy=0; idy<N+1; ++idy) {
	    
	    UdataFlux(i  ,j  , dofMapF(idx,idy,0,ivar_out)) = flux[idy];
	    
	  }
	  
	} // end for icomp
	
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
  void operator()(const typename Kokkos::Impl::enable_if<dim_==3, int>::type& index) const
  {

    const int isize = this->params.isize;
    const int jsize = this->params.jsize;
    const int ksize = this->params.ksize;

    // local cell index
    int i,j,k;
    index2coord(index,i,j,k,isize,jsize,ksize);

    solution_values_t sol;
    flux_values_t     flux;
    
    const Kokkos::Array<int,3> ivar_in_list  = {IU,  IV,  IW};
    const Kokkos::Array<int,3> ivar_out_list = {IGU, IGV, IGW};

    // loop over cell DoF's
    if (dir == IX) {

      for (int idz=0; idz<N; ++idz) {
	for (int idy=0; idy<N; ++idy) {
	  
	  // for each variables
	  for (int icomp = 0; icomp<dim; ++icomp) {

	    const int ivar_in  = ivar_in_list[icomp];
	    const int ivar_out = ivar_out_list[icomp];
	    
	    // get solution values vector along X direction
	    for (int idx=0; idx<N; ++idx) {
	      
	      sol[idx] =
		UdataSol(i,j,k, dofMapS(idx,idy,idz,ivar_in)) /
		UdataSol(i,j,k, dofMapS(idx,idy,idz,ID));
	      
	    }
	    
	    // interpolate at flux points for this given variable
	    this->sol2flux_vector(sol, flux);
	    	    
	    // copy back interpolated value
	    for (int idx=0; idx<N+1; ++idx) {
	      
	      UdataFlux(i,j,k, dofMapF(idx,idy,idz,ivar_out)) = flux[idx];
	      
	    }
	    
	  } // end for icomp
	  
	} // end for idy
      } // end for idz
      
    } // end for dir IX

    // loop over cell DoF's
    if (dir == IY) {

      for (int idz=0; idz<N; ++idz) {
	for (int idx=0; idx<N; ++idx) {
	  
	  // for each variables
	  for (int icomp = 0; icomp<dim; ++icomp) {

	    const int ivar_in  = ivar_in_list[icomp];
	    const int ivar_out = ivar_out_list[icomp];
	    
	    // get solution values vector along Y direction
	    for (int idy=0; idy<N; ++idy) {
	      
	      sol[idy] =
		UdataSol(i,j,k, dofMapS(idx,idy,idz,ivar_in)) /
		UdataSol(i,j,k, dofMapS(idx,idy,idz,ID));
	    
	    }
	    
	    // interpolate at flux points for this given variable
	    this->sol2flux_vector(sol, flux);
	    	    
	    // copy back interpolated value
	    for (int idy=0; idy<N+1; ++idy) {
	      
	      UdataFlux(i,j,k, dofMapF(idx,idy,idz,ivar_out)) = flux[idy];
	      
	    }
	  
	  } // end for icomp
	
	} // end for idx
      } // end for idz

    } // end for dir IY

    // loop over cell DoF's
    if (dir == IZ) {

      for (int idy=0; idy<N; ++idy) {
	for (int idx=0; idx<N; ++idx) {
	  
	  // for each variables
	  for (int icomp = 0; icomp<dim; ++icomp) {

	    const int ivar_in  = ivar_in_list[icomp];
	    const int ivar_out = ivar_out_list[icomp];
	    
	    // get solution values vector along Y direction
	    for (int idz=0; idz<N; ++idz) {
	      
	      sol[idz] =
		UdataSol(i,j,k, dofMapS(idx,idy,idz,ivar_in)) /
		UdataSol(i,j,k, dofMapS(idx,idy,idz,ID));
	      
	    }
	    
	    // interpolate at flux points for this given variable
	    this->sol2flux_vector(sol, flux);
	    
	    // copy back interpolated value
	    for (int idz=0; idz<N+1; ++idz) {
	      
	      UdataFlux(i,j,k, dofMapF(idx,idy,idz,ivar_out)) = flux[idz];
	      
	    }
	  
	  } // end for icomp
	
	} // end for idx
      } // end for idz

    } // end for dir IZ

  } // end operator () - 3d
  
  DataArray UdataSol, UdataFlux;

}; // class Interpolate_velocities_Sol2Flux_Functor

//! typedef used in functors performing average at cell borders
using var_index_t = Kokkos::Array<int,16>;

/*************************************************/
/*************************************************/
/*************************************************/
/**
 * This functor takes as an input a flux array, and perform
 * a simple average (half sum) of some components of the input data array
 * at cell borders.
 *
 * The primary use of this functor is to average velocity components, but can be other fields.
 *
 * Please note that velocity components in the flux in,out array must be addressed through 
 * IGU, IGV, IGW defined in enum class VarIndexGrad2d and VarIndexGrad3d.
 *
 */
template<int dim, int N, int dir>
class Average_component_at_cell_borders_Functor : public SDMBaseFunctor<dim,N> {

public:
  using typename SDMBaseFunctor<dim,N>::DataArray;
  using typename SDMBaseFunctor<dim,N>::HydroState;
  
  using SDMBaseFunctor<dim,N>::IGU;
  using SDMBaseFunctor<dim,N>::IGV;
  using SDMBaseFunctor<dim,N>::IGW;
  using SDMBaseFunctor<dim,N>::IGT;

  static constexpr auto dofMapF = DofMapFlux<dim,N,dir>;

  /**
   * \param[in] params
   * \param[in] sdm_geom
   * \param[in,out] UdataFlux a flux array
   */
  Average_component_at_cell_borders_Functor(HydroParams         params,
					    SDM_Geometry<dim,N> sdm_geom,
					    DataArray           UdataFlux,
					    int                 nbvar,
					    var_index_t         var_index) :
    SDMBaseFunctor<dim,N>(params,sdm_geom),
    UdataFlux(UdataFlux),
    nbvar(nbvar),
    var_index()
  {};

  
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
    const int jsize = this->params.jsize;

    // local cell index
    int i,j;
    index2coord(index,i,j,isize,jsize);

    real_t dataL, dataR, data_average;
    
    // =========================
    // ========= DIR X =========
    // =========================
    if (dir == IX) {

      // avoid ghost cells
      if (i>0 and i<isize) {

	// just deals with the left cell border
	for (int idy=0; idy<N; ++idy) {

	  for (int iv = 0; iv<nbvar; iv++) {
	    int ivar = var_index[iv];
	    
	    dataL = UdataFlux(i-1,j, dofMapF(N,idy,0,ivar));
	    dataR = UdataFlux(i  ,j, dofMapF(0,idy,0,ivar));
	    data_average = 0.5*(dataL + dataR);
	    UdataFlux(i-1,j, dofMapF(N,idy,0,ivar)) = data_average;
	    UdataFlux(i  ,j, dofMapF(0,idy,0,ivar)) = data_average;
	  }
	  
	} // end for idy

      } // end ghost cells guard
      
    } // end dir IX
    
    // =========================
    // ========= DIR Y =========
    // =========================
    if (dir == IY) {

      // avoid ghost cells
      if (j>0 and j<jsize) {

	// just deals with the left cell border
	for (int idx=0; idx<N; ++idx) {

	  for (int iv = 0; iv<nbvar; iv++) {
	    int ivar = var_index[iv];
	    
	    dataL = UdataFlux(i,j-1, dofMapF(idx,N,0,ivar));
	    dataR = UdataFlux(i,j  , dofMapF(idx,0,0,ivar));
	    data_average = 0.5*(dataL + dataR);
	    UdataFlux(i,j-1, dofMapF(idx,N,0,ivar)) = data_average;
	    UdataFlux(i,j  , dofMapF(idx,0,0,ivar)) = data_average;
	  }
	  
	} // end for idy

      } // end ghost cells guard
      
    } // end dir IY
    
  } // end operator() - 2d
  
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
    const int ksize = this->params.ksize;

    // local cell index
    int i,j,k;
    index2coord(index,i,j,k,isize,jsize,ksize);

    real_t dataL, dataR, data_average;
    
    // =========================
    // ========= DIR X =========
    // =========================
    if (dir == IX) {

      // avoid ghost cells
      if (i>0 and i<isize) {

	// just deals with the left cell border
	for (int idz=0; idz<N; ++idz) {
	  for (int idy=0; idy<N; ++idy) {

	    for (int iv = 0; iv<nbvar; iv++) {
	      int ivar = var_index[iv];

	      dataL = UdataFlux(i-1,j, k, dofMapF(N,idy,idz,ivar));
	      dataR = UdataFlux(i  ,j, k, dofMapF(0,idy,idz,ivar));
	      data_average = 0.5*(dataL + dataR);
	      UdataFlux(i-1,j, k, dofMapF(N,idy,idz,ivar)) = data_average;
	      UdataFlux(i  ,j, k, dofMapF(0,idy,idz,ivar)) = data_average;
	    } // end for ivar
	  
	  } // end for idy
	} // end for idz
	
      } // end ghost cells guard
      
    } // end dir IX

    // =========================
    // ========= DIR Y =========
    // =========================
    if (dir == IY) {

      // avoid ghost cells
      if (j>0 and j<jsize) {

	// just deals with the left cell border
	for (int idz=0; idz<N; ++idz) {
	  for (int idx=0; idx<N; ++idx) {

	    for (int iv = 0; iv<nbvar; iv++) {
	      int ivar = var_index[iv];

	      dataL = UdataFlux(i,j-1, k, dofMapF(idx,N,idz,ivar));
	      dataR = UdataFlux(i,j  , k, dofMapF(idx,0,idz,ivar));
	      data_average = 0.5*(dataL + dataR);
	      UdataFlux(i,j-1, k, dofMapF(idx,N,idz,ivar)) = data_average;
	      UdataFlux(i,j  , k, dofMapF(idx,0,idz,ivar)) = data_average;
	    } // end for ivar
	  
	  } // end for idx
	} // end for idz
	
      } // end ghost cells guard
      
    } // end dir IY

    // =========================
    // ========= DIR Z =========
    // =========================
    if (dir == IZ) {

      // avoid ghost cells
      if (k>0 and k<ksize) {

	// just deals with the left cell border
	for (int idy=0; idy<N; ++idy) {
	  for (int idx=0; idx<N; ++idx) {

	    for (int iv = 0; iv<nbvar; iv++) {
	      int ivar = var_index[iv];

	      dataL = UdataFlux(i,j, k-1, dofMapF(idx,idy,N,ivar));
	      dataR = UdataFlux(i,j, k  , dofMapF(idx,idy,0,ivar));
	      data_average = 0.5*(dataL + dataR);
	      UdataFlux(i,j, k-1, dofMapF(idx,idy,N,ivar)) = data_average;
	      UdataFlux(i,j, k  , dofMapF(idx,idy,0,ivar)) = data_average;
	    } // end for ivar
	  
	  } // end for idx
	} // end for idy
	
      } // end ghost cells guard
      
    } // end dir IZ

  } // end operator() - 3d
    
  DataArray UdataFlux;

  /** number of variables to average */
  int nbvar;

  /** array of integer, to index component of UdataFlux to be averaged */
  var_index_t var_index;
  
}; // class Average_component_at_cell_borders_Functor

/*************************************************/
/*************************************************/
/*************************************************/
/**
 * This functor takes as an input a fluxes data array (only IU,IV, IW are used)
 * containing velocity at flux points, supposed to be continuous at flux points (i.e.
 * an array ouput by function Average_component_at_cell_borders_Functor)
 * and perform interpolation of velocity gradients at solution points.
 *
 * Only one direction of gradient is considered (specified as template parameter dir).
 * 
 * \todo we will need to modify this functor to include as an option the possibility to compute
 * gradient of temperature.
 */
template<int dim, int N, int dir,
	 Interpolation_type_t itype=INTERPOLATE_DERIVATIVE>
class Interp_grad_velocity_at_SolutionPoints_Functor : public SDMBaseFunctor<dim,N> {
  
public:
  using typename SDMBaseFunctor<dim,N>::DataArray;
  using typename SDMBaseFunctor<dim,N>::solution_values_t;
  using typename SDMBaseFunctor<dim,N>::flux_values_t;

  using SDMBaseFunctor<dim,N>::IGU;
  using SDMBaseFunctor<dim,N>::IGV;
  using SDMBaseFunctor<dim,N>::IGW;
  using SDMBaseFunctor<dim,N>::IGT;

  static constexpr auto dofMapS = DofMap<dim,N>;
  static constexpr auto dofMapF = DofMapFlux<dim,N,dir>;

  /**
   *
   * \param[in] UdataFlux array containing velocity at flux points along given direction dir (template parameter); UdataFlux must have the same memory layout as FUgrad in SolverHydroSDM
   *
   * \param[out] UdataSol velocity gradient along dir at solution points
   *
   */
  Interp_grad_velocity_at_SolutionPoints_Functor(HydroParams         params,
						 SDM_Geometry<dim,N> sdm_geom,
						 DataArray           UdataFlux,
						 DataArray           UdataSol) :
    SDMBaseFunctor<dim,N>(params,sdm_geom),
    UdataFlux(UdataFlux),
    UdataSol(UdataSol)
  {};

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
    const int jsize = this->params.jsize;

    // rescale factor for derivative
    real_t rescale = 1.0/this->params.dx;
    if (dir == IY)
      rescale = 1.0/this->params.dy;
    
    // local cell index
    int i,j;
    index2coord(index,i,j,isize,jsize);

    solution_values_t sol;
    flux_values_t     flux;
    
    const Kokkos::Array<int,2> index_list  = {IGU, IGV};

    // loop over cell DoF's
    if (dir == IX) {

      for (int idy=0; idy<N; ++idy) {

	// for each variables
	for (int ivar = 0; ivar<dim; ++ivar ) {

	  const int index_var = index_list[ivar];
	  
	  // get values at flux point along X direction
	  for (int idx=0; idx<N+1; ++idx)
	    flux[idx] = UdataFlux(i  ,j  , dofMapF(idx,idy,0,index_var));
	    
	  // interpolate at flux points for this given variable
	  if (itype==INTERPOLATE_SOLUTION)
	    this->flux2sol_vector(flux, sol);
	  else
	    this->flux2sol_derivative_vector(flux,sol,rescale);
	  
	  // copy back interpolated value
	  for (int idx=0; idx<N; ++idx)
	    UdataSol(i  ,j  , dofMapS(idx,idy,0,index_var)) = sol[idx];
	  
	} // end for ivar
	
      } // end for idy

    } // end for dir IX

    // loop over cell DoF's
    if (dir == IY) {

      for (int idx=0; idx<N; ++idx) {

	// for each variables
	for (int ivar=0; ivar<dim; ++ivar ) {
	
	  const int index_var = index_list[ivar];

	  // get values at flux point along Y direction
	  for (int idy=0; idy<N+1; ++idy)
	    flux[idy] = UdataFlux(i  ,j  , dofMapF(idx,idy,0,index_var));
	    	  
	  // interpolate at flux points for this given variable
	  if (itype==INTERPOLATE_SOLUTION)
	    this->flux2sol_vector(flux, sol);
	  else
	    this->flux2sol_derivative_vector(flux,sol,rescale);
	  
	  // copy back interpolated value
	  for (int idy=0; idy<N; ++idy)
	    UdataSol(i  ,j  , dofMapS(idx,idy,0,index_var)) += sol[idy];
	  
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
  void operator()(const typename Kokkos::Impl::enable_if<dim_==3, int>::type& index) const
  {

    const int isize = this->params.isize;
    const int jsize = this->params.jsize;
    const int ksize = this->params.ksize;

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
    
    const Kokkos::Array<int,3> index_list  = { IGU, IGV, IGW};

    // loop over cell DoF's
    if (dir == IX) {

      for (int idz=0; idz<N; ++idz) {
	for (int idy=0; idy<N; ++idy) {
	  
	  // for each variables
	  for (int ivar = 0; ivar<dim; ++ivar) {
	    
	    const int index_var = index_list[ivar];
	    
	    // get values at flux point along X direction
	    for (int idx=0; idx<N+1; ++idx)
	      flux[idx] = UdataFlux(i,j,k, dofMapF(idx,idy,idz,index_var));
	  
	    // interpolate at flux points for this given variable
	    if (itype==INTERPOLATE_SOLUTION)
	      this->flux2sol_vector(flux, sol);
	    else
	      this->flux2sol_derivative_vector(flux,sol,rescale);
	    
	    // copy back interpolated value
	    for (int idx=0; idx<N; ++idx)
	      UdataSol(i,j,k, dofMapS(idx,idy,idz,index_var)) += sol[idx];
	    
	  } // end for ivar
	  
	} // end for idy
      } // end for idz

    } // end for dir IX

    // loop over cell DoF's
    if (dir == IY) {

      for (int idz=0; idz<N; ++idz) {
	for (int idx=0; idx<N; ++idx) {

	  // for each variables
	  for (int ivar=0; ivar<dim; ++ivar) {
	    
	    const int index_var = index_list[ivar];
	    
	    // get values at flux point along Y direction
	    for (int idy=0; idy<N+1; ++idy)
	      flux[idy] = UdataFlux(i,j,k, dofMapF(idx,idy,idz,index_var));
	    
	    // interpolate at flux points for this given variable
	    if (itype==INTERPOLATE_SOLUTION)
	      this->flux2sol_vector(flux, sol);
	    else
	      this->flux2sol_derivative_vector(flux,sol,rescale);
	    
	    // copy back interpolated value
	    for (int idy=0; idy<N; ++idy)
	      UdataSol(i,j,k, dofMapS(idx,idy,idz,index_var)) += sol[idy];
	    
	  } // end for ivar
	  
	} // end for idx
      } // end for idz

    } // end for dir IY

    // loop over cell DoF's
    if (dir == IZ) {

      for (int idy=0; idy<N; ++idy) {
	for (int idx=0; idx<N; ++idx) {

	  // for each variables
	  for (int ivar=0; ivar<dim; ++ivar) {
	    
	    const int index_var = index_list[ivar];
	    
	    // get values at flux point along Y direction
	    for (int idz=0; idz<N+1; ++idz)
	      flux[idz] = UdataFlux(i,j,k, dofMapF(idx,idy,idz,index_var));
	    
	    // interpolate at flux points for this given variable
	    if (itype==INTERPOLATE_SOLUTION)
	      this->flux2sol_vector(flux, sol);
	    else
	      this->flux2sol_derivative_vector(flux,sol,rescale);
	    
	    // copy back interpolated value
	    for (int idz=0; idz<N; ++idz)
	      UdataSol(i,j,k, dofMapS(idx,idy,idz,index_var)) += sol[idz];
	    
	  } // end for ivar
	  
	} // end for idx
      } // end for idy

    } // end for dir IZ

  } // end operator () - 3d
  
  DataArray UdataFlux, UdataSol;

}; // Interp_grad_velocity_at_SolutionPoints_Functor


/*************************************************/
/*************************************************/
/*************************************************/
/**
 * This functor takes as an input velocity gradients
 * at solution points and perform interpolation at flux points. 
 * What happends at cell borders is the subject
 * of an another functor : Average_component_gradient_at_cell_borders_functor
 *
 * Please note that velocity components in the flux out array must be addressed through 
 * indexes defined in enum class VarIndexGrad2d and VarIndexGrad3d.
 *
 * \tparam dim is dimension (2 or 3)
 * \tparam N is SDM order (number of solution point per dimension.
 * \tparam dir specifies the flux points direction (IX, IY or IZ)
 * \tparam dir_grad specifies the gradient direction (IX, IY or IZ)
 * dir_grad controls the indexes used to adress the output array.
 *
 */
template<int dim, int N, int dir, int dir_grad>
class Interpolate_velocity_gradients_Sol2Flux_Functor : public SDMBaseFunctor<dim,N> {

public:
  using typename SDMBaseFunctor<dim,N>::DataArray;
  using typename SDMBaseFunctor<dim,N>::solution_values_t;
  using typename SDMBaseFunctor<dim,N>::flux_values_t;
  
  using SDMBaseFunctor<dim,N>::IGU;
  using SDMBaseFunctor<dim,N>::IGV;
  using SDMBaseFunctor<dim,N>::IGW;

  using SDMBaseFunctor<dim,N>::IGUX;
  using SDMBaseFunctor<dim,N>::IGVX;
  using SDMBaseFunctor<dim,N>::IGWX;

  using SDMBaseFunctor<dim,N>::IGUY;
  using SDMBaseFunctor<dim,N>::IGVY;
  using SDMBaseFunctor<dim,N>::IGWY;

  using SDMBaseFunctor<dim,N>::IGUZ;
  using SDMBaseFunctor<dim,N>::IGVZ;
  using SDMBaseFunctor<dim,N>::IGWZ;

  //using SDMBaseFunctor<dim,N>::IGT;

  static constexpr auto dofMapS = DofMap<dim,N>;
  static constexpr auto dofMapF = DofMapFlux<dim,N,dir>;

  /**
   * \param[in] UdataSol array of a velocity gradient array at solution points (either Ugradx_v, Ugrady_v or Ugradz_v).
   * \param[out] UdataFlux array of velocity gradients interpolated at flux points.
   *
   * Please note that velocity components in the flux out array must be addressed through 
   * indexes defined in enum class VarIndexGrad2d and VarIndexGrad3d.
   *
   * This means UdataFlux should have been allocated like FUgrad (see class SolverHydroSDM).
   */
  Interpolate_velocity_gradients_Sol2Flux_Functor(HydroParams         params,
						  SDM_Geometry<dim,N> sdm_geom,
						  DataArray           UdataSol,
						  DataArray           UdataFlux) :
    SDMBaseFunctor<dim,N>(params,sdm_geom),
    UdataSol(UdataSol),
    UdataFlux(UdataFlux)
  {};
  
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
    const int jsize = this->params.jsize;

    // local cell index
    int i,j;
    index2coord(index,i,j,isize,jsize);

    solution_values_t sol;
    flux_values_t     flux;

    // loop over cell DoF's
    if (dir == IX) {

      const Kokkos::Array<int,2> ivar_in_list  = {IGU,  IGV};
      Kokkos::Array<int,2> ivar_out_list;
      if (dir_grad == IX) ivar_out_list = {IGUX, IGVX};
      if (dir_grad == IY) ivar_out_list = {IGUY, IGVY};
      
      for (int idy=0; idy<N; ++idy) {

	// for each velocity components
	for (int icomp = 0; icomp<dim; ++icomp) {

	  const int ivar_in  = ivar_in_list[icomp];
	  const int ivar_out = ivar_out_list[icomp];
	  
	  // get solution values vector along X direction
	  for (int idx=0; idx<N; ++idx) {

	    // divide momentum by density to obtain velocity
	    sol[idx] =
	      UdataSol(i  ,j  , dofMapS(idx,idy,0,ivar_in)) /
	      UdataSol(i  ,j  , dofMapS(idx,idy,0,ID));

	  }
	  
	  // interpolate at flux points for this given variable
	  this->sol2flux_vector(sol, flux);
	  
	  // copy back interpolated value
	  for (int idx=0; idx<N+1; ++idx) {
	    
	    UdataFlux(i  ,j  , dofMapF(idx,idy,0,ivar_out)) = flux[idx];
	    
	  } // end for idx
	  
	} // end for icomp
	
      } // end for idy

    } // end for dir IX

    // loop over cell DoF's
    if (dir == IY) {

      const Kokkos::Array<int,2> ivar_in_list  = {IGU,  IGV};
      Kokkos::Array<int,2> ivar_out_list;
      if (dir_grad == IX) ivar_out_list = {IGUX, IGVX};
      if (dir_grad == IY) ivar_out_list = {IGUY, IGVY};

      for (int idx=0; idx<N; ++idx) {

	// for each variables
	for (int icomp = 0; icomp<dim; ++icomp) {

	  const int ivar_in  = ivar_in_list[icomp];
	  const int ivar_out = ivar_out_list[icomp];
	
	  // get solution values vector along Y direction
	  for (int idy=0; idy<N; ++idy) {
	  
	    sol[idy] =
	      UdataSol(i  ,j  , dofMapS(idx,idy,0,ivar_in)) /
	      UdataSol(i  ,j  , dofMapS(idx,idy,0,ID));
	    
	  }
	  
	  // interpolate at flux points for this given variable
	  this->sol2flux_vector(sol, flux);
	  
	  // copy back interpolated value
	  for (int idy=0; idy<N+1; ++idy) {
	    
	    UdataFlux(i  ,j  , dofMapF(idx,idy,0,ivar_out)) = flux[idy];
	    
	  }
	  
	} // end for icomp
	
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
  void operator()(const typename Kokkos::Impl::enable_if<dim_==3, int>::type& index) const
  {

    const int isize = this->params.isize;
    const int jsize = this->params.jsize;
    const int ksize = this->params.ksize;

    // local cell index
    int i,j,k;
    index2coord(index,i,j,k,isize,jsize,ksize);

    solution_values_t sol;
    flux_values_t     flux;
    
    // loop over cell DoF's
    if (dir == IX) {

      const Kokkos::Array<int,3> ivar_in_list  = {IGU,  IGV,  IGW};
      Kokkos::Array<int,3> ivar_out_list;
      if (dir_grad == IX) ivar_out_list = {IGUX, IGVX, IGWX};
      if (dir_grad == IY) ivar_out_list = {IGUY, IGVY, IGWY};
      if (dir_grad == IZ) ivar_out_list = {IGUZ, IGVZ, IGWZ};

      for (int idz=0; idz<N; ++idz) {
	for (int idy=0; idy<N; ++idy) {
	  
	  // for each variables
	  for (int icomp = 0; icomp<dim; ++icomp) {

	    const int ivar_in  = ivar_in_list[icomp];
	    const int ivar_out = ivar_out_list[icomp];
	    
	    // get solution values vector along X direction
	    for (int idx=0; idx<N; ++idx) {
	      
	      sol[idx] =
		UdataSol(i,j,k, dofMapS(idx,idy,idz,ivar_in)) /
		UdataSol(i,j,k, dofMapS(idx,idy,idz,ID));
	      
	    }
	    
	    // interpolate at flux points for this given variable
	    this->sol2flux_vector(sol, flux);
	    	    
	    // copy back interpolated value
	    for (int idx=0; idx<N+1; ++idx) {
	      
	      UdataFlux(i,j,k, dofMapF(idx,idy,idz,ivar_out)) = flux[idx];
	      
	    }
	    
	  } // end for icomp
	  
	} // end for idy
      } // end for idz
      
    } // end for dir IX

    // loop over cell DoF's
    if (dir == IY) {

      const Kokkos::Array<int,3> ivar_in_list  = {IGU,  IGV,  IGW};
      Kokkos::Array<int,3> ivar_out_list;
      if (dir_grad == IX) ivar_out_list = {IGUX, IGVX, IGWX};
      if (dir_grad == IY) ivar_out_list = {IGUY, IGVY, IGWY};
      if (dir_grad == IZ) ivar_out_list = {IGUZ, IGVZ, IGWZ};

      for (int idz=0; idz<N; ++idz) {
	for (int idx=0; idx<N; ++idx) {
	  
	  // for each variables
	  for (int icomp = 0; icomp<dim; ++icomp) {

	    const int ivar_in  = ivar_in_list[icomp];
	    const int ivar_out = ivar_out_list[icomp];
	    
	    // get solution values vector along Y direction
	    for (int idy=0; idy<N; ++idy) {
	      
	      sol[idy] =
		UdataSol(i,j,k, dofMapS(idx,idy,idz,ivar_in)) /
		UdataSol(i,j,k, dofMapS(idx,idy,idz,ID));
	    
	    }
	    
	    // interpolate at flux points for this given variable
	    this->sol2flux_vector(sol, flux);
	    	    
	    // copy back interpolated value
	    for (int idy=0; idy<N+1; ++idy) {
	      
	      UdataFlux(i,j,k, dofMapF(idx,idy,idz,ivar_out)) = flux[idy];
	      
	    }
	  
	  } // end for icomp
	
	} // end for idx
      } // end for idz

    } // end for dir IY

    // loop over cell DoF's
    if (dir == IZ) {

      const Kokkos::Array<int,3> ivar_in_list  = {IGU,  IGV,  IGW};
      Kokkos::Array<int,3> ivar_out_list;
      if (dir_grad == IX) ivar_out_list = {IGUX, IGVX, IGWX};
      if (dir_grad == IY) ivar_out_list = {IGUY, IGVY, IGWY};
      if (dir_grad == IZ) ivar_out_list = {IGUZ, IGVZ, IGWZ};

      for (int idy=0; idy<N; ++idy) {
	for (int idx=0; idx<N; ++idx) {
	  
	  // for each variables
	  for (int icomp = 0; icomp<dim; ++icomp) {

	    const int ivar_in  = ivar_in_list[icomp];
	    const int ivar_out = ivar_out_list[icomp];
	    
	    // get solution values vector along Y direction
	    for (int idz=0; idz<N; ++idz) {
	      
	      sol[idz] =
		UdataSol(i,j,k, dofMapS(idx,idy,idz,ivar_in)) /
		UdataSol(i,j,k, dofMapS(idx,idy,idz,ID));
	      
	    }
	    
	    // interpolate at flux points for this given variable
	    this->sol2flux_vector(sol, flux);
	    
	    // copy back interpolated value
	    for (int idz=0; idz<N+1; ++idz) {
	      
	      UdataFlux(i,j,k, dofMapF(idx,idy,idz,ivar_out)) = flux[idz];
	      
	    }
	  
	  } // end for icomp
	
	} // end for idx
      } // end for idz

    } // end for dir IZ

  } // end operator () - 3d
  
  DataArray UdataSol, UdataFlux;
  
}; // class Interpolate_velocity_gradients_Sol2Flux_Functor

} // namespace sdm

#endif // SDM_INTERPOLATE_VISCOUS_FUNCTORS_H_
