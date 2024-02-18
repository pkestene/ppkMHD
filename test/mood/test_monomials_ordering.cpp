#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "mood/monomials_ordering.h"
#include "mood/monomials_print_utils.h"

int
main(int argc, char * argv[])
{

  int order = 4;
  if (argc > 1)
    order = atoi(argv[1]);

  std::cout << "Print all multivariate monomials up to order in dimension 2\n";
  mood::print_all_monomials<2>(order);

  std::cout << "Print all multivariate monomials up to order in dimension 3\n";
  mood::print_all_monomials<3>(order);

  // print all multivariate monomials up to order in dimension 4
  // mood::print_all_monomials<4>(order);

  return 0;
}
