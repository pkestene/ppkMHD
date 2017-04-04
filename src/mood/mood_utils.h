#ifndef MOOD_UTILS_H_
#define MOOD_UTILS_H_

// shared
#include "shared/real_type.h"

// mood
#include "mood/Matrix.h"
#include "mood/Stencil.h"
#include "mood/MonomialMap.h"
#include "mood/GeometricTerms.h"

namespace mood {

/**
 * Fill the Matrix with the geometric terms (for regular cartesian grid).
 *
 * \param[in,out] mat matrix to fill
 * \param[in] stencil
 * \param[in] monomialMap
 * 
 * \tparam dim dimension (either 2 or 3)
 * \tparam degree polynomial degree
 */
template<int dim, int degree>
void fill_geometry_matrix(Matrix& mat,
			  Stencil stencil,
			  const MonomialMap<dim,degree>& monomialMap,
			  std::array<real_t,3> dxyz)
{
  
  // create geometric terms struct
  real_t dx = dxyz[0];
  real_t dy = dxyz[1];
  real_t dz = dxyz[2];
  mood::GeometricTerms geomTerms(dx,dy,dz);

  int i=0;

  if (dim==2) {
    
    // loop over stencil point
    for (int is = 0; is<stencil.stencilSize; ++is) {
      
      int x = stencil.offsets_h(is,0);
      int y = stencil.offsets_h(is,1);
      
      // avoid stencil center
      if (x != 0 or y != 0) {
	
	// loop over monomial
	for (int j = 0; j<mat.n; ++j) {
	  // stencil point
	  // get monomial exponent for j+1 (to avoid the constant term)
	  int n = monomialMap.data_h(j+1,0);
	  int m = monomialMap.data_h(j+1,1);
	  
	  mat(i,j) = geomTerms.eval_hat(x,y,n,m);
	}
	
	++i;
      } 
    } // end for is

  } else {

    // loop over stencil point
    for (int is = 0; is<stencil.stencilSize; ++is) {
      
      int x = stencil.offsets_h(is,0);
      int y = stencil.offsets_h(is,1);
      int z = stencil.offsets_h(is,2);
      
      // avoid stencil center
      if (x != 0 or y != 0 or z != 0) {
	
	// loop over monomial
	for (int j = 0; j<mat.n; ++j) {
	  // stencil point
	  // get monomial exponent for j+1 (to avoid the constant term)
	  int n = monomialMap.data_h(j+1,0);
	  int m = monomialMap.data_h(j+1,1);
	  int l = monomialMap.data_h(j+1,2);
	  
	  mat(i,j) = geomTerms.eval_hat(x,y,z,
					n,m,l);
	}
	
	++i;
      }
      
    } // end for

  } // end 2d / 3d
  
} // fill_geometry_matrix

} // namespace mood

#endif // MOOD_UTILS_H_
