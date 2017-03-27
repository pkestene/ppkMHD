#ifndef MOOD_FUNCTORS_H_
#define MOOD_FUNCTORS_H_

#include <array>

#include "shared/kokkos_shared.h"
#include "shared/HydroParams.h"
#include "shared/HydroState.h"

#include "mood/mood_shared.h"
#include "mood/Polynomial.h"
#include "mood/MoodBaseFunctor.h"
#include "mood/QuadratureRules.h"

namespace mood {

// =======================================================================
// =======================================================================
/**
 * Compute MOOD fluxes.
 * 
 * Please note:
 * - DataArray and HydroState are typedef'ed in MoodBaseFunctor
 * - FluxData_z may or may not be allocated (depending dim==2 or 3).
 *
 * stencilId must be known at compile time, so that stencilSize is too.
 */
template<int dim,
	 int degree,
	 STENCIL_ID stencilId>
class ComputeFluxesFunctor : public MoodBaseFunctor<dim,degree>
{
    
public:
  using typename MoodBaseFunctor<dim,degree>::DataArray;
  using typename PolynomialEvaluator<dim,degree>::coefs_t;
  
  /**
   * Constructor for 2D/3D.
   */
  ComputeFluxesFunctor(DataArray        Udata,
		       DataArray        FluxData_x,
		       DataArray        FluxData_y,
		       DataArray        FluxData_z,
		       HydroParams      params,
		       Stencil          stencil,
		       mood_matrix_pi_t mat_pi) :
    MoodBaseFunctor<dim,degree>(params),
    Udata(Udata),
    FluxData_x(FluxData_x),
    FluxData_y(FluxData_y),
    FluxData_z(FluxData_z),
    stencil(stencil),
    mat_pi(mat_pi)
  {};

  ~ComputeFluxesFunctor() {};

  //! functor for 2d 
  template<int dim_ = dim>
  KOKKOS_INLINE_FUNCTION
  void operator()(const typename Kokkos::Impl::enable_if<dim_==2, int>::type& index)  const
  {

    const int isize = this->params.isize;
    const int jsize = this->params.jsize;
    const int ghostWidth = this->params.ghostWidth;

    const real_t dx = this->params.dx;
    const real_t dy = this->params.dy;
    
    int i,j;
    index2coord(index,i,j,isize,jsize);
    
    // rhs is sized upon stencil, just remove central point
    Kokkos::Array<real_t,stencil_size-1> rhs;

    // rhs for neighbor cell (accross an x-face, y-face or z-face)
    //Kokkos::Array<real_t,stencil_size-1> rhs_n;
    
    
    if(j >= ghostWidth && j < jsize-ghostWidth+1  &&
       i >= ghostWidth && i < isize-ghostWidth+1 ) {

      // retrieve neighbors data for ID, and build rhs
      int irhs = 0;
      for (int is=0; is<stencil_size; ++is) {
	int x = stencil.offsets(is,0);
	int y = stencil.offsets(is,1);
	if (x != 0 or y != 0) {
	  rhs[irhs] = Udata(i+x,j+y,ID) - Udata(i,j,ID);
	  irhs++;
	}	
      } // end for is

      // retrieve reconstruction polynomial coefficients in current cell
      coefs_t coefs_c;
      coefs_c[0] = Udata(i,j,ID);
      for (int icoef=0; icoef<mat_pi.dimension_0(); ++icoef) {
	real_t tmp = 0;
	for (int k=0; k<mat_pi.dimension_1(); ++k) {
	  tmp += mat_pi(icoef,k) * rhs[k];
	}
	coefs_c[icoef+1] = tmp;
      }

      // reconstruct Udata on the left face along X direction
      // for each quadrature points
      if (nbQuadraturePoints==1) {
	//int x = QUADRATURE_LOCATION_2D_N1_X_M[0][IX];
	//int y = QUADRATURE_LOCATION_2D_N1_X_M[0][IY];
      }

      FluxData_x(i,j,ID) = this->eval(-0.5*dx, 0.0   ,coefs_c);
      FluxData_y(i,j,ID) = this->eval( 0.0   ,-0.5*dy,coefs_c);
      
    } // end if
    
  } // end functor 2d

  //! functor for 3d 
  template<int dim_ = dim>
  KOKKOS_INLINE_FUNCTION
  void operator()(const typename Kokkos::Impl::enable_if<dim_==3, int>::type& index) const
  {
    const int isize = this->params.isize;
    const int jsize = this->params.jsize;
    const int ksize = this->params.ksize;
    const int ghostWidth = this->params.ghostWidth;

    const real_t dx = this->params.dx;
    const real_t dy = this->params.dy;
    const real_t dz = this->params.dz;

    int i,j,k;
    index2coord(index,i,j,k,isize,jsize,ksize);

    // rhs is sized upon stencil, just remove central point
    Kokkos::Array<real_t,stencil_size-1> rhs;
    
    if(k >= ghostWidth && k < ksize - ghostWidth+1 &&
       j >= ghostWidth && j < jsize - ghostWidth+1 &&
       i >= ghostWidth && i < isize - ghostWidth+1) {

      // retrieve neighbors data for ID, and build rhs
      int irhs = 0;
      for (int is=0; is<stencil_size; ++is) {
	int x = stencil.offsets(is,0);
	int y = stencil.offsets(is,1);
	int z = stencil.offsets(is,2);
	if (x != 0 or y != 0 or z != 0) {
	  rhs[irhs] = Udata(i+x,j+y,k+z,ID) - Udata(i,j,k,ID);
	  irhs++;
	}	
      } // end for is

      // retrieve reconstruction polynomial coefficients in current cell
      coefs_t coefs_c;
      coefs_c[0] = Udata(i,j,k,ID);
      for (int icoef=0; icoef<mat_pi.dimension_0(); ++icoef) {
	real_t tmp = 0;
	for (int ik=0; ik<mat_pi.dimension_1(); ++ik) {
	  tmp += mat_pi(icoef,ik) * rhs[ik];
	}
	coefs_c[icoef+1] = tmp;
      }

      // reconstruct Udata on the left face along X direction
      // for each quadrature points
      if (nbQuadraturePoints==1) {
	//int x = QUADRATURE_LOCATION_3D_N1_X_M[0][IX];
	//int y = QUADRATURE_LOCATION_3D_N1_X_M[0][IY];
	//int z = QUADRATURE_LOCATION_3D_N1_X_M[0][IZ];
      }

      FluxData_x(i,j,k,ID) = this->eval(-0.5*dx, 0.0   , 0.0   , coefs_c);
      FluxData_y(i,j,k,ID) = this->eval( 0.0   ,-0.5*dy, 0.0   , coefs_c);
      FluxData_z(i,j,k,ID) = this->eval( 0.0   , 0.0   ,-0.5*dz, coefs_c);

      
    }
    
  }  // end functor 3d
  
  DataArray        Udata;
  DataArray        FluxData_x, FluxData_y, FluxData_z;

  Stencil          stencil;
  mood_matrix_pi_t mat_pi;

  // get the number of cells in stencil
  static constexpr int stencil_size = STENCIL_SIZE[stencilId];

  // get the number of quadrature point per face corresponding to this stencil
  static constexpr int nbQuadraturePoints = QUADRATURE_NUM_POINTS[stencilId];
  
}; // class ComputeFluxesFunctor

// =======================================================================
// =======================================================================
/**
 * Compute MOOD polynomial coefficients.
 * 
 * Please note:
 * - DataArray and HydroState are typedef'ed in MoodBaseFunctor
 *
 * stencilId must be known at compile time, so that stencilSize is too.
 */
template<int dim,
	 int degree,
	 STENCIL_ID stencilId>
class ComputeReconstructionPolynomialFunctor : public MoodBaseFunctor<dim,degree>
{
    
public:
  using typename MoodBaseFunctor<dim,degree>::DataArray;
  using typename PolynomialEvaluator<dim,degree>::coefs_t;

  //! total number of coefficients in the polynomial
  static const int ncoefs =  mood::binomial<dim+degree,dim>();

  /**
   * Constructor for 2D/3D.
   */
  ComputeReconstructionPolynomialFunctor(DataArray                    Udata,
					 std::array<DataArray,ncoefs> polyCoefs,
					 HydroParams                  params,
					 Stencil                      stencil,
					 mood_matrix_pi_t             mat_pi) :
    MoodBaseFunctor<dim,degree>(params),
    Udata(Udata),
    polyCoefs(polyCoefs),
    stencil(stencil),
    mat_pi(mat_pi)
  {};

  ~ComputeReconstructionPolynomialFunctor() {};

  //! functor for 2d 
  template<int dim_ = dim>
  KOKKOS_INLINE_FUNCTION
  void operator()(const typename Kokkos::Impl::enable_if<dim_==2, int>::type& index)  const
  {

    const int isize = this->params.isize;
    const int jsize = this->params.jsize;
    const int ghostWidth = this->params.ghostWidth;

    const real_t dx = this->params.dx;
    const real_t dy = this->params.dy;

    const real_t nbvar = this->params.nbvar;
    
    int i,j;
    index2coord(index,i,j,isize,jsize);
    
    // rhs is sized upon stencil, just remove central point
    Kokkos::Array<real_t,stencil_size-1> rhs;

    if(j >= ghostWidth-1 && j < jsize-ghostWidth+1  &&
       i >= ghostWidth-1 && i < isize-ghostWidth+1 ) {

      for (int ivar=0; ivar<nbvar; ++ivar) {
	
	// retrieve neighbors data for variable ivar, and build rhs
	int irhs = 0;
	for (int is=0; is<stencil_size; ++is) {
	  int x = stencil.offsets(is,0);
	  int y = stencil.offsets(is,1);
	  if (x != 0 or y != 0) {
	    rhs[irhs] = Udata(i+x,j+y,ivar) - Udata(i,j,ivar);
	    irhs++;
	  }	
	} // end for is
	
	// retrieve reconstruction polynomial coefficients in current cell
	coefs_t coefs_c;
	coefs_c[0] = Udata(i,j,ivar);
	for (int icoef=0; icoef<mat_pi.dimension_0(); ++icoef) {
	  real_t tmp = 0;
	  for (int ik=0; ik<mat_pi.dimension_1(); ++ik) {
	    tmp += mat_pi(icoef,ik) * rhs[ik];
	  }
	  coefs_c[icoef+1] = tmp;
	}

	// copy back results on device memory
	for (int icoef=0; icoef<ncoefs; ++icoef)
	  polyCoefs[icoef](i,j,ivar) = coefs_c[icoef];

      } // end for ivar
      
    } // end if
    
  } // end functor 2d

  //! functor for 3d 
  template<int dim_ = dim>
  KOKKOS_INLINE_FUNCTION
  void operator()(const typename Kokkos::Impl::enable_if<dim_==3, int>::type& index) const
  {
    const int isize = this->params.isize;
    const int jsize = this->params.jsize;
    const int ksize = this->params.ksize;
    const int ghostWidth = this->params.ghostWidth;

    const real_t dx = this->params.dx;
    const real_t dy = this->params.dy;
    const real_t dz = this->params.dz;

    const real_t nbvar = this->params.nbvar;

    int i,j,k;
    index2coord(index,i,j,k,isize,jsize,ksize);

    // rhs is sized upon stencil, just remove central point
    Kokkos::Array<real_t,stencil_size-1> rhs;
    
    if(k >= ghostWidth && k < ksize - ghostWidth+1 &&
       j >= ghostWidth && j < jsize - ghostWidth+1 &&
       i >= ghostWidth && i < isize - ghostWidth+1) {

      for (int ivar=0; ivar<nbvar; ++ivar) {

	// retrieve neighbors data for ivar, and build rhs
	int irhs = 0;
	for (int is=0; is<stencil_size; ++is) {
	  int x = stencil.offsets(is,0);
	  int y = stencil.offsets(is,1);
	  int z = stencil.offsets(is,2);
	  if (x != 0 or y != 0 or z != 0) {
	    rhs[irhs] = Udata(i+x,j+y,k+z,ivar) - Udata(i,j,k,ivar);
	    irhs++;
	  }	
	} // end for is

	// retrieve reconstruction polynomial coefficients in current cell
	coefs_t coefs_c;
	coefs_c[0] = Udata(i,j,k,ivar);
	for (int icoef=0; icoef<mat_pi.dimension_0(); ++icoef) {
	  real_t tmp = 0;
	  for (int ik=0; ik<mat_pi.dimension_1(); ++ik) {
	    tmp += mat_pi(icoef,ik) * rhs[ik];
	  }
	  coefs_c[icoef+1] = tmp;
	}
	
	// copy back results on device memory
	for (int icoef=0; icoef<ncoefs; ++icoef)
	  polyCoefs[icoef](i,j,k,ivar) = coefs_c[icoef];
	
      } // end for ivar
      
    } // end if i,j,k
    
  }  // end functor 3d
  
  DataArray                    Udata;
  std::array<DataArray,ncoefs> polyCoefs;

  Stencil          stencil;
  mood_matrix_pi_t mat_pi;

  // get the number of cells in stencil
  static constexpr int stencil_size = STENCIL_SIZE[stencilId];

  // get the number of quadrature point per face corresponding to this stencil
  static constexpr int nbQuadraturePoints = QUADRATURE_NUM_POINTS[stencilId];
  
}; // class ComputeReconstructionPolynomialFunctor

} // namespace mood

#endif // MOOD_FUNCTORS_H_
