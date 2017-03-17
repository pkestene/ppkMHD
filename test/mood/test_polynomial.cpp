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
constexpr int dim = 3;

// highest degree / order of the polynomial
constexpr int order = 2;

// use to initialize data for polynomial
constexpr int ncoefs = mood::binomial<order+dim,order>();



using scalar_t = Kokkos::View<double[1]>;
using scalar_host_t = Kokkos::View<double[1]>::HostMirror;

using coefs_t = Kokkos::Array<real_t,ncoefs>;
using Polynomial_t = mood::Polynomial<dim,order>;

/**
 * A dummy functor to test computation on device with class polynomial.
 */
class TestPolynomialFunctor {

public:

  scalar_t data;
  mood::MonomialMap& monomialMap;
  Kokkos::Array<real_t,dim> eval_point;
  
  TestPolynomialFunctor(scalar_t data,
			mood::MonomialMap& monomialMap,
			Kokkos::Array<real_t,dim> eval_point) :
    data(data),
    monomialMap(monomialMap),
    eval_point(eval_point) {};
  ~TestPolynomialFunctor() {};


  KOKKOS_INLINE_FUNCTION
  void operator()(const int& i) const
  {

    coefs_t coefs;
    for (int i=0; i<ncoefs; ++i)
      coefs[i] = 1.0*i;

    Polynomial_t polynomial(monomialMap, coefs);

    data(0) = polynomial.eval(eval_point);
    
  }
  
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
  mood::MonomialMap monomialMap(dim,order);

  if (dim == 2) {
    for (int i = 0; i<monomialMap.Ncoefs; ++i) {
      
      int e[2] = {monomialMap.data_h(i,0),
		  monomialMap.data_h(i,1)};
      std::cout << "    {";
      std::cout << e[0] << "," << e[1] << "," << "},";
      std::cout << "   // " << "X^" << e[0] << " * " << "Y^" << e[1] << "\n";
      
    }
  } else {
    for (int i = 0; i<monomialMap.Ncoefs; ++i) {
      
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
  Kokkos::Array<real_t,dim> p;
  if (dim==2) {
    p[0] =  1.0;
    p[1] = -1.0;
  } else {
    p[0] =  1.12;
    p[1] =  0.02;
    p[2] = -3.1;
  }
  
  // compute on device
  TestPolynomialFunctor f(data, monomialMap, p);
  Kokkos::parallel_for(1,f);

  Kokkos::deep_copy(data_h,data);
  std::cout << "result on device: " << data_h(0) << "\n";
  
  // compute on host
  coefs_t coefs;
  for (int i=0; i<ncoefs; ++i)
    coefs[i] = 1.0*i;
  
  Polynomial_t polynomial(monomialMap, coefs);

  if (dim == 2) {
    for (int i = 0; i<monomialMap.Ncoefs; ++i) {
      
      int e[2] = {monomialMap.data_h(i,0),
		  monomialMap.data_h(i,1)};
      std::cout << "    {";
      std::cout << e[0] << "," << e[1] << "," << "},";
      std::cout << "   // " << "X^" << e[0] << " * " << "Y^" << e[1] << "\n";
      
    }
  } else {
    for (int i = 0; i<monomialMap.Ncoefs; ++i) {
      
      int e[3] = {monomialMap.data_h(i,0),
		  monomialMap.data_h(i,1),
		  monomialMap.data_h(i,2)};
      // std::cout << coefs[i] *
      // 	mood::power(p[0],e[0]) *
      // 	mood::power(p[1],e[1]) *
      // 	mood::power(p[2],e[2]) << "    {";
      std::cout << "    {";
      std::cout << e[0] << "," << e[1] << "," << e[2] << "},";
      std::cout << "   // " << "X^" << e[0] << " * " << "Y^" << e[1] << " * " << "Z^" << e[2] << "\n";
      
    }
  }

  
  double result_on_host = polynomial.eval(p);
  std::cout << "result on host:   " << result_on_host << "\n";
  
  Kokkos::finalize();

  return EXIT_SUCCESS;
  
}
