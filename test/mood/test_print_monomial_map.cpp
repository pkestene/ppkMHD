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
template <int dim, int degree>
class TestMonomialMapFunctor
{

public:
  typename mood::MonomialMap<dim, degree>::MonomMap results;
  typename mood::MonomialMap<dim, degree>::MonomMap monomialMap;

  TestMonomialMapFunctor(typename mood::MonomialMap<dim, degree>::MonomMap results,
                         typename mood::MonomialMap<dim, degree>::MonomMap monomialMap)
    : results(results)
    , monomialMap(monomialMap){};
  ~TestMonomialMapFunctor(){};

  KOKKOS_INLINE_FUNCTION
  void
  operator()(const int & i) const
  {

    for (int icoef = 0; icoef < monomialMap.extent(0); ++icoef)
    {
      results(icoef, 0) = monomialMap(icoef, 0);
      results(icoef, 1) = monomialMap(icoef, 1);
      results(icoef, 2) = monomialMap(icoef, 2);
    }
  }

}; // TestMonomialMapFunctor

int
main(int argc, char * argv[])
{

  Kokkos::initialize(argc, argv);

  // dim is the number of variable in the multivariate polynomial representation
  constexpr int dim = 3;

  // highest degree of the polynomial
  constexpr int degree = 4;

  // print all multivariate monomials up to order in dimension dim
  if (dim == 2)
  {
    mood::print_all_monomials<2>(degree);
    mood::print_all_monomials_map<2>(degree);
  }
  else if (dim == 3)
  {
    mood::print_all_monomials<3>(degree);
    mood::print_all_monomials_map<3>(degree);
  }
  else
  {
    std::cerr << "Not implemented or not tested !\n";
  }


  /*
   * test MonomialMap structure
   */
  std::cout << "############################\n";
  std::cout << " Testing MonomialMap Struct \n";
  std::cout << "############################\n";

  // dim = 3, degree = 4
  mood::MonomialMap<dim, degree> monomialMap;

  for (int i = 0; i < mood::MonomialMap<dim, degree>::ncoefs; ++i)
  {
    int e[3] = { monomialMap.data_h(i, 0), monomialMap.data_h(i, 1), monomialMap.data_h(i, 2) };

    std::cout << "    {";
    std::cout << e[0] << "," << e[1] << "," << e[2] << "},";
    std::cout << "   // " << "X^" << e[0] << " * " << "Y^" << e[1] << " * " << "Z^" << e[2] << "\n";
  }

  std::cout << "####################################\n";
  std::cout << "Testing MonomialMap Struct on Device\n";
  std::cout << "####################################\n";
  int                                      ncoefs = mood::binom(dim + degree, dim);
  mood::MonomialMap<dim, degree>::MonomMap data_device =
    mood::MonomialMap<dim, degree>::MonomMap("data_device");
  mood::MonomialMap<dim, degree>::MonomMap::HostMirror data_host =
    Kokkos::create_mirror_view(data_device);

  TestMonomialMapFunctor<dim, degree> f(data_device, monomialMap.data);
  Kokkos::parallel_for(1, f);

  // retrieve results
  Kokkos::deep_copy(data_host, data_device);

  for (int i = 0; i < mood::MonomialMap<dim, degree>::ncoefs; ++i)
  {
    int e[3] = { data_host(i, 0), data_host(i, 1), data_host(i, 2) };

    std::cout << "    {";
    std::cout << e[0] << "," << e[1] << "," << e[2] << "},";
    std::cout << "   // " << "X^" << e[0] << " * " << "Y^" << e[1] << " * " << "Z^" << e[2] << "\n";
  }

  Kokkos::finalize();

  return EXIT_SUCCESS;
}
