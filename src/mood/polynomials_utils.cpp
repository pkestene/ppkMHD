#include "mood/polynomials_utils.h"

namespace mood {

// some template specialization.

/*
 * Number of coefficients of a bivariate polynomial.
 */
template<>
int get_number_of_coefficients<2>(unsigned int order) {
  return (order+1)*(order+2)/2;
} // get_number_of_coefficients<2>

/*
 * Number of coefficients of a trivariate polynomial.
 */
template<>
int get_number_of_coefficients<3>(unsigned int order) {
  return (order+1)*(order+2)*(order+3)/6;
} // get_number_of_coefficients<3>


} // namespace mood
