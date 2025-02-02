#ifndef GEOMETRIC_TERMS_H_
#define GEOMETRIC_TERMS_H_

#include <array>

#include "shared/real_type.h"

namespace mood
{

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
class GeometricTerms
{

public:
  /**
   *
   */
  GeometricTerms(real_t dx, real_t dy, real_t dz);
  ~GeometricTerms();

  //! size of a cell (assumes here regular cartesian mesh)
  real_t dx, dy, dz;

  /**
   * computes volume average of x^n y^m z^l inside
   * cell i (chosen to be the origin).
   *
   * In 2D:
   * \f$ \overline{x^n y^m}_i = \frac{1}{V_i} \int_{\mathcal{V}_i} (x-x_i)^n (y-y_i)^m dv \f$
   * with x_i = 0 and y_i = 0.
   *
   */
  real_t
  eval_moment(int i, int j, int n, int m);

  /**
   * computes volume average of x^n y^m z^l inside
   * cell i (chosen to be the origin).
   *
   * In 3D:
   * \f$ \overline{x^n y^m z^l}_i = \frac{1}{V_i} \int_{\mathcal{V}_i} (x-x_i)^n (y-y_i)^m (z-z_i)^l
   * dv \f$ with x_i = 0, y_i = 0 and z_i = 0.
   *
   */
  real_t
  eval_moment(int i, int j, int k, int n, int m, int l);

  /**
   * In 2D, this is formula from Ollivier-Gooch, 1997
   *
   * For structured grid, it returns the following
   * \f$ \widehat{x^n y^m}_{i,j} =\frac{1}{V_j} \int_{\mathcal{V}_j} \left( (x-x_j)+(x_j-x_i)
   * \right)^n \left( (y-y_j)+(y_j-y_i) \right)^m  \dif v  - \overline{x^n y^m}_i \f$ which can be
   * computed exactly on regular cartesian grid.
   *
   * For unstructured grid, it can be developed into
   *
   * \f$  \widehat{x^n y^m}_{i,j} = \sum_{a=0}^{n} \sum_{b=0}^{m} \binom{n}{a} \binom{m}{b}
   * (x_j-x_i)^a (y_j-y_i)^b \overline{x^{n-a} y^{m-b}}_j \; - \; \overline{x^n y^m}_i \f$
   *
   *
   * \param[in] xj coordinate (integers) of the target point.
   * \param[in] n
   * \param[in] m
   *
   * \return \f$  \widehat{x^n y^m}_{i,j} \f$
   */
  real_t
  eval_hat(int i, int j, int n, int m);

  /**
   * In 3D the formula is slightly adapted from the 2D version:
   *
   * By definition, we need to compute
   * \f$ \widehat{x^n y^m z^l}_{i,j} = \frac{1}{V_j} \int_{\mathcal{V}_j} \left( (x-x_j)+(x_j-x_i)
   * \right)^n \left( (y-y_j)+(y_j-y_i) \right)^m \left( (z-z_j)+(z_j-z_i) \right)^l  \dif v  -
   * \overline{x^n y^m z^l}_i\f$ which can be obtained analytically for a structured cartesian grid:
   *
   * \f$ \widehat{x^n y^m z^l}_{i,j} = ... \f$
   *
   * \param[in] xj coordinate (integers) of the target point.
   * \param[in] n
   * \param[in] m
   * \param[in] l
   *
   * \return \f$  \widehat{x^n y^m z^l}_{i,j} \f$
   *
   */
  real_t
  eval_hat(int i, int j, int k, int n, int m, int l);


}; // class GeometricTerms

} // namespace mood

#endif // GEOMETRIC_TERMS_H_
