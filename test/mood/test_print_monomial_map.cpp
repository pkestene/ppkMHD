/**
 * This executable is used to print on screen a map between an index
 * (spaning the total number of monomials) and a vector of exponents 
 * in the corresponding monomial.
 */

#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "shared/kokkos_shared.h"

#include "mood/monomials_ordering.h"
#include "mood/monomials_print_utils.h"

#include "mood/MonomialMap.h"

/**
 * A dummy functor to test computation on device with class MonomialMap.
 */
class TestMonomialMapFunctor {

public:
  mood::MonomialMap::MonomMap results;  
  mood::MonomialMap::MonomMap monomialMap;

  TestMonomialMapFunctor(mood::MonomialMap::MonomMap results,
			 mood::MonomialMap::MonomMap monomialMap) :
    results(results),
    monomialMap(monomialMap) {};
  ~TestMonomialMapFunctor() {};

  KOKKOS_INLINE_FUNCTION
  void operator()(const int& i) const
  {

    for (int icoef = 0; icoef < monomialMap.dimension_0(); ++icoef) {
      results(icoef, 0) = monomialMap(icoef,0);
      results(icoef, 1) = monomialMap(icoef,1);
      results(icoef, 2) = monomialMap(icoef,2);
    }
  }
  
}; // TestMonomialMapFunctor

int main(int argc, char* argv[])
{

  Kokkos::initialize(argc, argv);
 
  // dim is the number of variable in the multivariate polynomial representation
  unsigned int dim=3;

  // highest degree / order of the polynomial
  int order = 4;
  
  if (argc>1)
    dim = atoi(argv[1]);
  if (argc>2)
    order = atoi(argv[2]);

  // print all multivariate monomials up to order in dimension dim
  if        (dim == 2) {
    mood::print_all_monomials<2>(order);
    mood::print_all_monomials_map<2>(order);
  } else if (dim == 3) {
    mood::print_all_monomials<3>(order);
    mood::print_all_monomials_map<3>(order);
  } else {
    std::cerr << "Not implemented or not tested !\n";
  }


  /*
   * test MonomialMap structure
   */
  std::cout << "############################\n";
  std::cout << "Testing MonomialMap Struct\n";
  std::cout << "############################\n";

  // dim = 3, degree = 4
  mood::MonomialMap monomialMap(dim,order);
  
  for (int i = 0; i<monomialMap.Ncoefs; ++i) {
    int e[3] = {monomialMap.data_h(i,0),
		monomialMap.data_h(i,1),
		monomialMap.data_h(i,2)};

    std::cout << "    {";
    std::cout << e[0] << "," << e[1] << "," << e[2] << "},";
    std::cout << "   // " << "X^" << e[0] << " * " << "Y^" << e[1] << " * " << "Z^" << e[2] << "\n";

  }

  std::cout << "####################################\n";
  std::cout << "Testing MonomialMap Struct on Device\n";
  std::cout << "####################################\n";
  int Ncoefs = mood::binom(dim+order,dim);
  mood::MonomialMap::MonomMap data_device = mood::MonomialMap::MonomMap("data_device",Ncoefs,dim);
  mood::MonomialMap::MonomMap::HostMirror data_host = Kokkos::create_mirror_view(data_device);

  TestMonomialMapFunctor f(data_device, monomialMap.data);
  Kokkos::parallel_for(1,f);

  // retrieve results
  Kokkos::deep_copy(data_host,data_device);

  for (int i = 0; i<monomialMap.Ncoefs; ++i) {
    int e[3] = {data_host(i,0),
  		data_host(i,1),
  		data_host(i,2)};

    std::cout << "    {";
    std::cout << e[0] << "," << e[1] << "," << e[2] << "},";
    std::cout << "   // " << "X^" << e[0] << " * " << "Y^" << e[1] << " * " << "Z^" << e[2] << "\n";

  }
  
  Kokkos::finalize();

  return EXIT_SUCCESS;
  
}
