#ifndef POLYNOMIALS_H_
#define POLYNOMIALS_H_

#include <array>       // for std::array
#include <type_traits> // for std::integral_constant
//#include "real_type.h"
#define real_t = double;


namespace ppkMHD {

/**
 * An utility function to compute an integer power of a real number.
 * 
 * See http://stackoverflow.com/questions/16443682/c-power-of-integer-template-meta-programming/16443849#16443849
 */
template<class T>
inline constexpr T power(const T x, std::integral_constant<T, 0>){
  return 1;
}

template<class T, int N>
inline constexpr T power(const T x, std::integral_constant<T, N>){
  return power(x, std::integral_constant<T, N-1>()) * x;
}

// template<int N, class T>
// inline constexpr T power(const T x)
// {
//     return power(x, std::integral_constant<T, N>());
// }

template<int N>
inline constexpr real_t power(const real_t x)
{
    return power(x, std::integral_constant<real_t, N>());
}


/**
 * Return number of coefficients in a n-dimensional polynomial.
 *
 * When template parameter is 1, we consider a regular polynomial.
 * When template parameter is 2, we consider a bivariate polynomial.
 * When template parameter is 3, we consider a trivariate polynomial.
 *
 * \param[in] order of polynomial
 */
template<unsigned int dim>
int get_number_of_coefficients(unsigned int order) {
  return order+1;
}

/**
 * Number of coefficients of a bivariate polynomial.
 */
template<>
int get_number_of_coefficients<2>(unsigned int order) {
  return (order+1)*(order+2)/2;
}

/**
 * Number of coefficients of a trivariate polynomial.
 */
template<>
int get_number_of_coefficients<3>(unsigned int order) {
  return (order+1)*(order+2)*(order+3)/6;
}

/**
 * A minimal data structure representing a bi-or-tri variate polynomial.
 *
 * This must be small as it will be used in a Kokkos kernel.
 */
template<unsigned int dim, unsigned int order>
class Polynomial {

private:
  int Ncoefs = get_number_of_coefficients<dim>(order);
  std::array<real_t,Ncoefs> coefs;

public:
  Polynomial(const std::array<real_t,Ncoefs>& coefs) : coefs(coefs) {};

  /** evaluate polynomial at a given point */
  real_t eval(real_t x, real_t y, real_t z);

  /** evaluate polynomial at a given point */
  real_t eval(real_t x, real_t y) {

    real_t result;
    int c = 0;
    
    if (dim == 2) {

      for (int d = 0; d <= order; ++d) {

	for (i=0; i<=d; i++) {

	  // result += coefs[c] * x^d-i * y^i
	  result += coefs[c] * power<d-i>(x) * power<i>(y);
	  
	}
	
      }

    } 
    
  };

  real_t getCoefs(int i) {
    return coefs[i];
  }
  
private:
  int Ncoefs = get_number_of_coefficients<dim>(order);
  std::array<real_t,Ncoefs> coefs;
    
  
}; // class Polynomial

/**
 * Just print a polynomial for checking.
 */
template<unsigned int dim, unsigned int order>
void print_polynomial(const Polynomial<dim,order>& poly)
{

  if (order == 2) {

    
    
  }

} // namespace ppkMHD

#endif // POLYNOMIALS_H_
