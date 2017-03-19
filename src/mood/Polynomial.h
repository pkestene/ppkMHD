#ifndef MOOD_POLYNOMIALS_H_
#define MOOD_POLYNOMIALS_H_

#include <array>       // for std::array
#include <type_traits> // for std::integral_constant
#include "shared/real_type.h"

#include "mood/polynomials_utils.h"
#include "mood/monomials_ordering.h"
#include "mood/MonomialMap.h"

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
 * A minimal data structure representing a bi-or-tri variate polynomial.
 *
 * This must be small as it will be used/created in a Kokkos kernel functor.
 */
template<unsigned int dim, unsigned int order>
class Polynomial {

public:
  //! total number of coefficients in the polynomial
  static const int Ncoefs =  binomial<dim+order,dim>();

  //! typedef for the coefs array
  using coefs_t = Kokkos::Array<real_t,Ncoefs>;

  //! typedef for coordinates points
  using point_t = Kokkos::Array<real_t,dim>;
  
private:
  //! array containing the polynomial coefficients 
  Kokkos::Array<real_t,Ncoefs>& coefs;
  
public:
  /**
   * this is a map spanning all possibles monomials, and all possible
   * variables, such that monomials_map[i][j] gives the exponent of the
   * jth variables in the ith monomials.
   */
  const MonomialMap& monomialMap;
  
public:
  Polynomial(const MonomialMap& monomialMap,
	     coefs_t& coefs) :
    monomialMap(monomialMap),
    coefs(coefs) {};

  /** evaluate polynomial at a given point (2D) */
  KOKKOS_INLINE_FUNCTION
  real_t eval(real_t x, real_t y) const {

    real_t result = 0;
    int c = 0;
    
    if (dim == 2) {

      // span monomial orders
      for (int i = 0; i<Ncoefs; ++i) {
	int e[2] = {monomialMap.data(i,0),
		    monomialMap.data(i,1)};
	result += coefs[i] * power(x,e[0]) * power(y,e[1]);
      }

      
    } // end dim == 2

    return result;

  }; // eval 2D

  /** evaluate polynomial at a given point (3D) */
  KOKKOS_INLINE_FUNCTION
  real_t eval(real_t x, real_t y, real_t z) const {

    real_t result=0;
    int c=0;

    if (dim == 3) {

      // span all monomials in Graded Reverse Lexicographical order
      for (int i = 0; i<Ncoefs; ++i) {

	int e[3] = {monomialMap.data(i,0),
		    monomialMap.data(i,1),
		    monomialMap.data(i,2)};
	result += coefs[i] * power(x,e[0]) * power(y,e[1]) * power(z,e[2]);
	
      }
      
    }

    return result;
    
  }; // eval 3D

  /** evaluate polynomial at a given point (3D) */
  KOKKOS_INLINE_FUNCTION
  real_t eval(point_t p) const {

    real_t result=0;
    
    if (dim == 2) {

      real_t x = p[0]; 
      real_t y = p[1];
      
      int c = 0;
      
      // span monomial orders
      for (int i = 0; i<Ncoefs; ++i) {
	int e[2] = {monomialMap.data(i,0),
		    monomialMap.data(i,1)};
	result += coefs[i] * power(x,e[0]) * power(y,e[1]);
      }
      
    } // end dim == 2

    if (dim == 3) {

      real_t x = p[0]; 
      real_t y = p[1];
      real_t z = p[2];
      
      int c = 0;

      // span all monomials in Graded Reverse Lexicographical order
      for (int i = 0; i<Ncoefs; ++i) {

	int e[3] = {monomialMap.data(i,0),
		    monomialMap.data(i,1),
		    monomialMap.data(i,2)};
	result += coefs[i] * power(x,e[0]) * power(y,e[1]) * power(z,e[2]);
	
      }
      
    } // end dim == 3

    return result;
    
  }; // eval generique

  KOKKOS_INLINE_FUNCTION
  real_t getCoefs(int i) const {

    real_t tmp = 0.0;
    if (i>=0 and i<Ncoefs)
      tmp = coefs[i];
    
    return tmp;
    
  } // getCoefs

  /**
   * set i-th coefficients
   */
  KOKKOS_INLINE_FUNCTION
  void setCoefs(int i, real_t value) const {

    if (i>=0 and i<Ncoefs)
      coefs[i] = value;
    
    return;
    
  } // setCoefs

  /**
   * set polynomial coefficients based on monomial exponents :
   * 
   * \param[in] e exponent array (identifying a given monomial)
   * \param[in] value is the specified monomial coefficient.
   */
  KOKKOS_INLINE_FUNCTION
  void setCoefs(Kokkos::Array<int,3> e, real_t value) const {

    if (dim == 2) {

      for (int i = 0; i<Ncoefs; ++i) {
	if (monomialMap.data(i,0) == e[0] and
	    monomialMap.data(i,1) == e[1]) {
	  coefs[i] = value;
	}
      }
      
    } else if (dim == 3) {

      for (int i = 0; i<Ncoefs; ++i) {
	if (monomialMap.data(i,0) == e[0] and
	    monomialMap.data(i,1) == e[1] and
	    monomialMap.data(i,2) == e[2]) {
	  coefs[i] = value;
	}
      }

    }

    return;
    
  } // setCoefs
    
}; // class Polynomial

// initialize static member
//template<unsigned int dim, unsigned int order>
//const MonomialMap<dim,order> Polynomial<dim,order>::monomialMap;

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

#endif // MOOD_POLYNOMIALS_H_
