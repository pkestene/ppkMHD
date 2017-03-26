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
 * A minimal base functor for evaluating a bi-or-tri variate polynomial.
 *
 * Polynomial coefficients are supposed to be provided on a per thread basis.
 * Monomial map is shared, it is a Kokkos::View.
 */
template<unsigned int dim, unsigned int degree>
class PolynomialEvaluator {

public:
  //! total number of coefficients in the polynomial
  static const int Ncoefs =  binomial<dim+degree,dim>();

  //! typedef for the coefs array
  using coefs_t = Kokkos::Array<real_t,Ncoefs>;

  //! typedef for coordinates points
  using point_t = Kokkos::Array<real_t,dim>;
  
public:
  /**
   * this is a map spanning all possibles monomials, and all possible
   * variables, such that monomMa(i,j) gives the exponent of the
   * jth variables in the ith monomials.
   *
   * MonomialMap::MonomMap is a Kokkos::View type defined in class MonomialMap.
   */
  const MonomialMap<dim,degree> monomialMap;
  
public:
  PolynomialEvaluator() : monomialMap() {};

  virtual ~PolynomialEvaluator() {};
  
  /** evaluate polynomial at a given point (2D) */
  KOKKOS_INLINE_FUNCTION
  real_t eval(real_t x, real_t y, coefs_t coefs) const {

    real_t result = 0;
    
    if (dim == 2) {

      // span monomial degrees
      for (int i = 0; i<Ncoefs; ++i) {
	int e[2] = {monomialMap.data(i,0),
		    monomialMap.data(i,1)};
	result += coefs[i] * pow(x,e[0]) * pow(y,e[1]);
      }

      
    } // end dim == 2

    return result;

  }; // eval 2D

  /** evaluate polynomial at a given point (3D) */
  KOKKOS_INLINE_FUNCTION
  real_t eval(real_t x, real_t y, real_t z, coefs_t coefs) const {
    
    real_t result=0;
 
    if (dim == 3) {

      // span all monomials in Graded Reverse Lexicographical degree
      for (int i = 0; i<Ncoefs; ++i) {

	int e[3] = {monomialMap.data(i,0),
		    monomialMap.data(i,1),
		    monomialMap.data(i,2)};
	result += coefs[i] * pow(x,e[0]) * pow(y,e[1]) * pow(z,e[2]);
	
      }
      
    }

    return result;
    
  }; // eval 3D

  /** evaluate polynomial at a given point (3D) */
  KOKKOS_INLINE_FUNCTION
  real_t eval(point_t p, coefs_t coefs) const {

    real_t result=0;
    
    if (dim == 2) {

      real_t x = p[0]; 
      real_t y = p[1];
      
      // span monomial degrees
      for (int i = 0; i<Ncoefs; ++i) {
	int e[2] = {monomialMap.data(i,0),
		    monomialMap.data(i,1)};
	result += coefs[i] * pow(x,e[0]) * pow(y,e[1]);
      }
      
    } // end dim == 2

    if (dim == 3) {

      real_t x = p[0]; 
      real_t y = p[1];
      real_t z = p[2];
      
      // span all monomials in Graded Reverse Lexicographical degree
      for (int i = 0; i<Ncoefs; ++i) {

	int e[3] = {monomialMap.data(i,0),
		    monomialMap.data(i,1),
		    monomialMap.data(i,2)};
	result += coefs[i] * pow(x,e[0]) * pow(y,e[1]) * pow(z,e[2]);
	
      }
      
    } // end dim == 3

    return result;
    
  }; // eval generique
    
}; // class PolynomialEvaluator


/**
 * set polynomial coefficients based on monomial exponents :
 * 
 * \param[in,out] coefs arrary of coefficients, one per monomial
 * \param[in] MonomialMap member data_h contains the map between monomial and exponents
 * \param[in] e0 e1 exponents (identifying a given monomial)
 * \param[in] value is the specified monomial coefficient.
 *
 * \tparam ncoefs number of polynomial coefficients
 * \tparam degree polynomial degree
 */
template<int ncoefs, int degree>
void polynomial_setCoefs(Kokkos::Array<real_t,ncoefs>& coefs,
			 MonomialMap<2,degree> monomialMap,
			 int e0, int e1,
			 real_t value) {

  // we should assert here that ncoefs == monomialMap.ncoefs
  
  for (int i = 0; i<ncoefs; ++i) {
    // find the right location of the monomial identified by e0, e1
    if (monomialMap.data_h(i,0) == e0 and
	monomialMap.data_h(i,1) == e1) {
      coefs[i] = value;
    }
  }
  
  return;
  
} // polynomial_setCoefs

/**
 * set polynomial coefficients based on monomial exponents :
 * 
 * \param[in,out] coefs arrary of coefficients, one per monomial
 * \param[in] MonomialMap member data_h contains the map between monomial and exponents
 * \param[in] e0 e1 e2 exponents (identifying a given monomial)
 * \param[in] value is the specified monomial coefficient.
 *
 * \tparam ncoefs number of polynomial coefficients
 * \tparam degree polynomial degree
 */
template<int ncoefs, int degree>
void polynomial_setCoefs(Kokkos::Array<real_t,ncoefs>& coefs,
			 MonomialMap<3,degree>& monomialMap,
			 int e0, int e1, int e2,
			 real_t value) {

  // we should assert here that ncoefs == monomialMap.ncoefs

  for (int i = 0; i<ncoefs; ++i) {

    // find the right location of the monomial identified by e0, e1, e2
    if (monomialMap.data_h(i,0) == e0 and
	monomialMap.data_h(i,1) == e1 and
	monomialMap.data_h(i,2) == e2) {
      coefs[i] = value;
    }
    
  } // end for
  
  return;
  
} // polynomial_setCoefs

// initialize static member
//template<unsigned int dim, unsigned int degree>
//const MonomialMap<dim,degree> Polynomial<dim,degree>::monomialMap;


} // namespace mood

#endif // MOOD_POLYNOMIALS_H_
