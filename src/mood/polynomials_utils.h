/**
 * Some utilities for multivariate polynomials.
 * 
 */
#ifndef MOOD_POLYNOMIALS_UTILS_H_
#define MOOD_POLYNOMIALS_UTILS_H_

#include "mood/Binomial.h"

namespace mood {

/**
 * Return number of coefficients in a n-dimensional (i.e. multivariate) polynomial
 * of a given order.
 *
 * When template parameter is 1, we consider a regular polynomial.
 * When template parameter is 2, we consider a bivariate polynomial.
 * When template parameter is 3, we consider a trivariate polynomial.
 *
 * The general formula is (dim+order)! / dim! / order!
 *
 * An elegant proof can be found here:
 * http://math.stackexchange.com/questions/380116/number-of-coefficients-of-multivariable-polynomial
 * which relies on "star and bars" arguments:
 * https://en.wikipedia.org/wiki/Stars_and_bars_%28combinatorics%29
 *
 * TODO : remove template specialization, implement the general formula as in 
 * boost/math/special_functions/binomial.hpp
 *
 * \param[in] order of polynomial
 */
template<unsigned int dim>
int get_number_of_coefficients(unsigned int order) {

  // default value for univariate polynomial
  return order+1;

} // get_number_of_coefficients

/**
 * Number of coefficients of a bivariate polynomial.
 */
template<>
int get_number_of_coefficients<2>(unsigned int order) {
  return (order+1)*(order+2)/2;
} // get_number_of_coefficients<2>

/**
 * Number of coefficients of a trivariate polynomial.
 */
template<>
int get_number_of_coefficients<3>(unsigned int order) {
  return (order+1)*(order+2)*(order+3)/6;
} // get_number_of_coefficients<3>


/**
 * The general formula is (dim+order)! / dim! / order! (Binomial coefficient).
 */
template<unsigned int dim, unsigned int order>
int get_number_of_coefficients() {

  // default value for univariate polynomial
  return binomial<dim+order,dim>();

} // get_number_of_coefficients

} // namespace mood

#endif // MOOD_POLYNOMIALS_UTILS_H_
