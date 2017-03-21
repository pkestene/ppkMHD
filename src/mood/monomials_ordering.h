/**
 * About multivariate monomials ordering:
 *
 * - https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-972-algebraic-techniques-and-semidefinite-optimization-spring-2006/lecture-notes/lecture_14.pdf
 *
 * - https://en.wikipedia.org/wiki/Monomial_order#Graded_reverse_lexicographic_order
 *
 * The routines defined here are helper to enumerate all multivariate monomials
 * up to a given order.
 */

#ifndef MONOMIALS_ORDERING_H_
#define MONOMIALS_ORDERING_H_

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <array>

namespace mood {

/**
 * This routine compute the vector of exponents of the next monomial using
 * the graded reverse lexical order (grlex).
 *
 * template parameter is dimension.
 * e.g. 
 * - dim=2 means  bi-variate polynomial.
 * - dim=3 means tri-variate polynomial.
 *
 * \param[in] e exponent vector
 *
 * e[0] exponent of X_0 (= X)
 * e[1] exponent of X_1 (= Y)
 *
 * Example of a tri-variate monomial:
 * e[] = {3,4,1} ==> X_0^e[0] * X_1^e[1] * X_2^e[2]
 *                   X_0      * X_1^4    * X_2^3 
 *                   X        * Y^4      * Z^3
 *
 *
 * This routine is slightly adapted from John Burkardt :
 * https://people.sc.fsu.edu/~jburkardt/cpp_src/polynomial/polynomial.html
 *
 * We just change exponent ordering:
 *
 * e = {e[0], e[1], e[2]}
 * e[0] is the least significant exponent (X_0 = X)
 * e[1] is                                (X_1 = Y)
 * e[2] is the most  significant exponent (X_2 = Z)
 *
 */
template<unsigned int dim>
//void mono_next_grlex (int e[dim] )
void mono_next_grlex (std::array<int,dim>& e)
{
  int i;
  int j;
  int t;

  //
  //  Find I, the index of the rightmost nonzero entry of exponent vector.
  //
  i = 0;
  for ( j = dim; 1 <= j; j-- ) {
    if ( 0 < e[dim-j] )
      {
	i = j;
	break;
      }
  }
  //
  //  set t = e(i)
  //  set e(i) to zero,
  //  increase e(i-1) by 1,
  //  increment e(dim-1) by t-1.
  //
  if ( i == 0 ) {

    // increment least significant exponent
    e[0] = 1;
    return;

  } else if ( i == 1 ) {

    // increment monomial order
    e[0] = e[0] + e[dim-1] + 1;
    e[dim-1] = 0;

  } else if ( i > 1 ) {

    t = e[dim-i];
      
    e[dim-i] = 0;
    e[dim-i+1] = e[dim-i+1] + 1;
    e[0] = e[0] + t - 1;
    
  }
  
  return;
  
} // mono_next_grlex

/**
 * This is the original monomial ordering routine from John Burkardt's website:
 * https://people.sc.fsu.edu/~jburkardt/cpp_src/polynomial/polynomial.html
 * 
 * \sa mono_next_grlex
 */
template<unsigned int dim>
void mono_next_grlex_orig (int e[dim] )
{
  int i;
  int j;
  int t;

  //
  //  Find I, the index of the rightmost nonzero entry of exponent vector.
  //
  i = 0;
  for ( j = dim; 1 <= j; j-- ) {
    if ( 0 < e[j-1] )
      {
	i = j;
	break;
      }
  }
  //
  //  set t = e(i)
  //  set e(i) to zero,
  //  increase e(i-1) by 1,
  //  increment e(dim-1) by t-1.
  //
  if ( i == 0 ) {

    // increment least significant exponent
    e[dim-1] = 1;
    return;

  } else if ( i == 1 ) {

    // increment monomial order
    e[dim-1] = e[dim-1] + e[0] + 1;
    e[0] = 0;

  } else if ( i > 1 ) {

    t = e[i-1];
      
    e[i-1] = 0;
    e[i-2] = e[i-2] + 1;
    e[dim-1] = e[dim-1] + t - 1;
    
  }
  
  return;
  
} // mono_next_grlex

} // namespace mood

#endif // MONOMIALS_ORDERING_H_
