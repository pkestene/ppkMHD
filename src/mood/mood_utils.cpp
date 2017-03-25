#include "mood/mood_utils.h"
#include "mood/GeometricTerms.h"

namespace mood {

/**
 * Fill the Matrix with the geometric terms (for regular cartesian grid).
 *
 * \param[in,out] mat matrix to fill
 * \param[in] stencil
 * \param[in] monomialMap
 */
void fill_geometry_matrix_2d(Matrix& mat,
			     Stencil stencil,
			     const MonomialMap& monomialMap,
			     std::array<real_t,3> dxyz)
{

  // create geometric terms struct
  real_t dx = dxyz[0];
  real_t dy = dxyz[1];
  real_t dz = dxyz[2];
  mood::GeometricTerms geomTerms(dx,dy,dz);

  int i=0;

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
  }

} // fill_geometry_matrix_2d

/**
 * Fill the Matrix with the geometric terms (for regular cartesian grid).
 *
 * \param[in,out] mat matrix to fill
 * \param[in] stencil
 * \param[in] monomialMap
 */
void fill_geometry_matrix_3d(Matrix& mat,
			     Stencil stencil,
			     const MonomialMap& monomialMap,
			     std::array<real_t,3> dxyz)
{

  // create geometric terms struct
  real_t dx = dxyz[0];
  real_t dy = dxyz[1];
  real_t dz = dxyz[2];
  mood::GeometricTerms geomTerms(dx,dy,dz);

  int i=0;

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
  }

} // fill_geometry_matrix_3d

} // namespace mood
