#ifndef GEOMETRIC_TERMS_H_
#define GEOMETRIC_TERMS_H_

#include <array>
#include "shared/real_type.h"

namespace mood {

/**
 * \class GeometricTerms GeometricTerms.h
 *
 * \brief This class is a helper to compute geometric terms found
 * in the 2D/3D MOOD numerical scheme. 
 *
 * The reconstruction polynomial
 * is the solution of a linear system whose coefficients are 
 * purely geometric.
 *
 * Most of the code written here is adapted from reference:
 * Ollivier-Gooch, Quasi-ENO schemes for unstructured meshes based
 * on unlimited data-dependent least-squares reconstruction, JCP, 
 * vol 133, 6-17, 1997.
 * http://www.sciencedirect.com/science/article/pii/S0021999196955849
 */
class GeometricTerms {

public:

  /**
   *
   */
  GeometricTerms(real_t dx,
		 real_t dy,
		 real_t dz);
  ~GeometricTerms();

  //! size of a cell (assumes here regular cartesian mesh)
  real_t dx, dy, dz;
  
  /**
   * computes volume average of x^n y^m z^l inside cell i.
   *
   * In 2D:
   * \f$ \overline{x^n y^m}_i = \frac{1}{V_i} \int_{\mathcal{V}_i} (x-x_i)^n (y-y_i)^m dv \f$
   *
   *
   */
  real_t eval_moment(real_t x,
		     real_t y,
		     int n,
		     int m);
  
  /**
   * computes volume average of x^n y^m z^l inside cell i.
   *
   * In 3D:
   * \f$ \overline{x^n y^m z^l}_i = \frac{1}{V_i} \int_{\mathcal{V}_i} (x-x_i)^n (y-y_i)^m (z-z_i)^l dv \f$
   *
   */
  real_t eval_moment(real_t x,
		     real_t y,
		     real_t z,
		     int n,
		     int m,
		     int l);

  /**
   * In 2D, this is formula from Ollivier-Gooch, 1997
   *
   * returns \f$ \widehat{x^n y^m}_{i,j} =\frac{1}{V_j} \int_{\mathcal{V}_j} \left( (x-x_j)+(x_j-x_i) \right)^n \left( (y-y_j)+(y_j-y_i) \right)^m  \dif v  - \overline{x^n y^m}_i \f$
   *
   * Its can be developped into
   *
   * \f$  \widehat{x^n y^m}_{i,j} = \sum_{a=0}^{n} \sum_{b=0}^{m} \binom{n}{a} \binom{m}{b} (x_j-x_i)^a (y_j-y_i)^b \overline{x^{n-a} y^{m-b}}_j \; - \; \overline{x^n y^m}_i
\f$
   */
  real_t eval_hat(const std::array<real_t,2>& xi,
		  const std::array<real_t,2>& xj,
		  int n, int m);

  /**
   * returns \f$ \widehat \f$
   */
  real_t eval_hat(const std::array<real_t,3>& xi,
		  const std::array<real_t,3>& xj,
		  int n, int m, int l);

  
}; // class GeometricTerms

} // namespace mood

#endif // GEOMETRIC_TERMS_H_