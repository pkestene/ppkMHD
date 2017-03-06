#ifndef POLYNOMIALS_H_
#define POLYNOMIALS_H_

#include <array>       // for std::array
#include <type_traits> // for std::integral_constant
#include "shared/real_type.h"

#include "mood/polynomials_utils.h"
#include "mood/monomials_ordering.h"

namespace mood {

/**
 * An utility function to compute an integer power of a real number.
 * 
 * See http://stackoverflow.com/questions/16443682/c-power-of-integer-template-meta-programming/16443849#16443849
 */
template<class T>
inline constexpr T power(const T base, unsigned int const exponent)
{
  // (parentheses not required in next line)
  return (exponent == 0) ? 1 :
    (exponent & 1) ? base * power(base, (exponent>>1)) * power(base, (exponent>>1)) :
    power(base, (exponent>>1))*power(base, (exponent>>1));
}

/**
 * MonomialMap structure holds an array with 2 entries:
 * - one for the monomial Id
 * - one for multivariate index
 *
 */
template<unsigned int dim, unsigned int order>
struct MonomialMap {

  //static const int Ncoefs = get_number_of_coefficients<dim>(order);
  static const int Ncoefs = binomial<dim+order,dim>();
  int data[Ncoefs][dim];

  /**
   * Default constructor build monomials map entries.
   */
  MonomialMap() {

    // exponent vector
    int e[dim];
    for (int i=0; i<dim; ++i) e[i] = 0;
    
    // d is the order, it will increase up to order
    int d = -1;
    
    int sum_e = 0;
    for (int i=0; i<dim; ++i) sum_e += e[i];

    int index = 0;
    
    // span all possible monomials
    while ( sum_e <= order ) {

      if (dim==2) {
	data[index][0] = e[0];
	data[index][1] = e[1];
      } else if (dim == 3) {
	data[index][0] = e[0];
	data[index][1] = e[1];
	data[index][2] = e[2];
      }
      
      // increment (in the sens of graded reverse lexicographic order)
      // the exponents vector representing a monomial x^e[0] * y^e[1] * z^[2]
      mono_next_grlex<dim>(e);
      
      // update sum of exponents
      sum_e = 0;
      for (int i=0; i<dim; ++i) sum_e += e[i];

      ++index;
      
    } // end while
    
  } // MonomialMap

}; // class MonomialMap


/**
 * A minimal data structure representing a bi-or-tri variate polynomial.
 *
 * This must be small as it will be used in a Kokkos kernel.
 */
template<unsigned int dim, unsigned int order>
class Polynomial {
  
private:
  static const int Ncoefs = get_number_of_coefficients<dim>(order);
  std::array<real_t,Ncoefs> coefs;

public:
  /**
   * this is a map spanning all possibles monomials, and all possible
   * variables, such that monomials_map[i][j] gives the exponent of the
   * jth variables in the ith monomials.
   */
  static const MonomialMap<dim,order> monomialMap;
  
public:
  Polynomial(const std::array<real_t,Ncoefs>& coefs) : coefs(coefs) {};

  /** evaluate polynomial at a given point */
  real_t eval(real_t x, real_t y, real_t z);

  /** evaluate polynomial at a given point */
  real_t eval(real_t x, real_t y) {

    real_t result;
    int c = 0;
    
    if (dim == 2) {

      // span monomial orders
      for (int d = 0; d <= order; ++d) {

	// for a given order, span all possible monomials
	for (int i=0; i<=d; i++) {

	  // result += coefs[c] * x^d-i * y^i
	  result += coefs[c] * power(x,d-i) * power(y,i);
	  
	}
	
      }

    } else if (dim == 3) {

      
      
    }
    
  };

  real_t getCoefs(int i) {
    return coefs[i];
  }
    
}; // class Polynomial

// static member
// template<unsigned int dim, unsigned int order>
// const int Polynomial<dim,order>::monomials_map[Polynomial<dim,order>::Ncoefs][dim] = {
  
// };



// /**
//  * Just print a polynomial for checking.
//  */
// template<unsigned int dim, unsigned int order>
// void print_polynomial(const Polynomial<dim,order>& poly)
// {

//   if (order == 2) {

    
    
//   }

// } // print_polynomial

} // namespace mood

#endif // POLYNOMIALS_H_
