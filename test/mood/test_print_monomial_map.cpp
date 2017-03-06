/**
 * This executable is used to print on screen a map between an index
 * (spaning the total number of monomials) and a vector of exponents 
 * in the corresponding monomial.
 */

#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "mood/monomials_ordering.h"
#include "mood/monomials_print_utils.h"

int main(int argc, char* argv[])
{

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

  return 0;
  
}
