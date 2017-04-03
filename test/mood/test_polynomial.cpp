/**
 * This executable is used to test polynomial class.
 */

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <array>

#include "mood/monomials_ordering.h"
#include "mood/monomials_print_utils.h"

#include "mood/Polynomial.h"
#include "shared/real_type.h"

// dim is the number of variable in the multivariate polynomial representation
constexpr int Dim = 3;

// highest degree / order of the polynomial
constexpr int Degree = 4;

using scalar_t = Kokkos::View<double[1]>;
using scalar_host_t = Kokkos::View<double[1]>::HostMirror;

real_t polynomial_eval(real_t x, real_t y,
		       Kokkos::View<int**,DEVICE>::HostMirror monomMap,
		       Kokkos::View<double*,Kokkos::OpenMP> coefs)  {
  
  real_t result = 0;
  
  // span monomial orders
  for (int i = 0; i<coefs.dimension_0(); ++i) {
    int e[2] = {monomMap(i,0),
		monomMap(i,1)};
    result += coefs[i] * pow(x,e[0]) * pow(y,e[1]);
  }
  
  return result;
  
}; // eval 2D

real_t polynomial_eval(real_t x, real_t y, real_t z,
		       Kokkos::View<int**,DEVICE>::HostMirror monomMap,
		       Kokkos::View<double*,Kokkos::OpenMP> coefs)  {
  
  real_t result = 0;
  
  // span monomial orders
  for (int i = 0; i<coefs.dimension_0(); ++i) {
    int e[3] = {monomMap(i,0),
		monomMap(i,1),
		monomMap(i,2)};
    result += coefs[i] * pow(x,e[0]) * pow(y,e[1]) * pow(z,e[2]);
  }
  
  return result;
  
}; // eval 3D


/**
 * A dummy functor to test computation on device with class polynomial.
 */
template<int dim, int degree>
class TestPolynomialFunctor {

public:
  //! total number of coefficients in the polynomial
  static const int ncoefs =  mood::binomial<dim+degree,dim>();

  //! number of polynomial coefficients
  using coefs_t = Kokkos::Array<real_t,ncoefs>;

  scalar_t data;
  typename mood::MonomialMap<dim,degree>::MonomMap monomMap;
  Kokkos::Array<real_t,dim> eval_point;
  
  TestPolynomialFunctor(scalar_t data,
			typename mood::MonomialMap<dim,degree>::MonomMap monomMap,
			Kokkos::Array<real_t,dim> eval_point) :
    data(data),
    monomMap(monomMap),
    eval_point(eval_point) {};
  ~TestPolynomialFunctor() {};


  template<int dim_ = dim>
  KOKKOS_INLINE_FUNCTION
  void operator()(const typename Kokkos::Impl::enable_if<dim_==2, int>::type& i) const
  {

    coefs_t coefs;
    for (int i=0; i<ncoefs; ++i)
      coefs[i] = 1.0*i;

    data(0) = polynomial_eval(eval_point[0], eval_point[1], coefs);
    
  }

  template<int dim_ = dim>
  KOKKOS_INLINE_FUNCTION
  void operator()(const typename Kokkos::Impl::enable_if<dim_==3, int>::type& i) const
  {

    coefs_t coefs;
    for (int i=0; i<ncoefs; ++i)
      coefs[i] = 1.0*i;

    data(0) = polynomial_eval(eval_point[0], eval_point[1], eval_point[2], coefs);
    
  }

  /** evaluate polynomial at a given point (2D) */
  KOKKOS_INLINE_FUNCTION
  real_t polynomial_eval(real_t x, real_t y, coefs_t coefs) const {

    real_t result = 0;
    
    if (dim == 2) {

      // span monomial orders
      for (int i = 0; i<ncoefs; ++i) {
	int e[2] = {monomMap(i,0),
		    monomMap(i,1)};
	result += coefs[i] * pow(x,e[0]) * pow(y,e[1]);
      }

      
    } // end dim == 2

    return result;

  }; // eval 2D

  /** evaluate polynomial at a given point (3D) */
  KOKKOS_INLINE_FUNCTION
  real_t polynomial_eval(real_t x, real_t y, real_t z, coefs_t coefs) const {

    real_t result=0;
 
    if (dim == 3) {

      // span all monomials in Graded Reverse Lexicographical order
      for (int i = 0; i<ncoefs; ++i) {

	int e[3] = {monomMap(i,0),
		    monomMap(i,1),
		    monomMap(i,2)};
	result += coefs[i] * pow(x,e[0]) * pow(y,e[1]) * pow(z,e[2]);
	
      }
      
    } // end dim == 3

    return result;
    
  }; // eval 3D

};

int main(int argc, char* argv[])
{

  Kokkos::initialize(argc, argv);

  /*
   * test class Polynomial.
   */
  std::cout << "############################\n";
  std::cout << "Testing class Polynomial    \n";
  std::cout << "############################\n";

  // create monomial map and print
  using mMap = mood::MonomialMap<Dim,Degree>;
  mMap monomialMap;

  if (Dim == 2) {
    for (int i = 0; i<mMap::ncoefs; ++i) {
      
      int e[2] = {monomialMap.data_h(i,0),
		  monomialMap.data_h(i,1)};
      std::cout << "    {";
      std::cout << e[0] << "," << e[1] << "},";
      std::cout << "   // " << "X^" << e[0] << " * " << "Y^" << e[1] << "\n";
      
    }
  } else {
    for (int i = 0; i<mMap::ncoefs; ++i) {
      
      int e[3] = {monomialMap.data_h(i,0),
		  monomialMap.data_h(i,1),
		  monomialMap.data_h(i,2)};
      std::cout << "    {";
      std::cout << e[0] << "," << e[1] << "," << e[2] << "},";
      std::cout << "   // " << "X^" << e[0] << " * " << "Y^" << e[1] << " * " << "Z^" << e[2] << "\n";
      
    }
  }
    
  // some dummy data
  scalar_t data("data");
  scalar_host_t data_h = Kokkos::create_mirror_view(data);
  data_h(0) = 1.0;
  Kokkos::deep_copy(data,data_h);

  // evaluation point
  Kokkos::Array<real_t,Dim> p;
  if (Dim==2) {
    p[0] =  1.04;
    p[1] = -1.65;
  } else {
    p[0] =  1.12;
    p[1] =  0.02;
    p[2] = -3.1;
  }
  
  // compute on device
  TestPolynomialFunctor<Dim,Degree> f(data, monomialMap.data, p);
  Kokkos::parallel_for(1,f);

  Kokkos::deep_copy(data_h,data);
  std::cout << "result on device: " << data_h(0) << "\n";
  
  // compute on host

  if (Dim == 2) {
    for (int i = 0; i<mMap::ncoefs; ++i) {
      
      int e[2] = {monomialMap.data_h(i,0),
		  monomialMap.data_h(i,1)};
      std::cout << "    {";
      std::cout << e[0] << "," << e[1] << "},";
      std::cout << "   // " << "X^" << e[0] << " * " << "Y^" << e[1] << "\n";
      
    }
  } else {
    for (int i = 0; i<mMap::ncoefs; ++i) {
      
      int e[3] = {monomialMap.data_h(i,0),
		  monomialMap.data_h(i,1),
		  monomialMap.data_h(i,2)};
      std::cout << "    {";
      std::cout << e[0] << "," << e[1] << "," << e[2] << "},";
      std::cout << "   // " << "X^" << e[0] << " * " << "Y^" << e[1] << " * " << "Z^" << e[2] << "\n";
      
    }
  }

  Kokkos::View<double*,Kokkos::OpenMP> coefs_view =
    Kokkos::View<double*,Kokkos::OpenMP>("coefs_view",monomialMap.ncoefs);
  for (int i = 0; i<monomialMap.ncoefs; ++i) {
    coefs_view(i) = 1.0*i;
  }
  
  if (Dim==2)
    std::cout << "result on host:   " << polynomial_eval(p[0],p[1],
							 monomialMap.data_h,
							 coefs_view) << "\n";
  else
    std::cout << "result on host:   " << polynomial_eval(p[0],p[1],p[2],
							 monomialMap.data_h,
							 coefs_view) << "\n";
  
  Kokkos::finalize();

  return EXIT_SUCCESS;
  
}
