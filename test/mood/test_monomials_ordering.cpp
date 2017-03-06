#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "mood/monomials_ordering.h"

int main(int argc, char* argv[])
{

  int order = 4;
  if (argc>1)
    order = atoi(argv[1]);

  // print all multivariate monomials up to order in dimension 2
  //print_all_monomials<2>(order);

  // print all multivariate monomials up to order in dimension 3
  print_all_monomials<3>(order);

  // print all multivariate monomials up to order in dimension 4
  //print_all_monomials<4>(order);
    
  return 0;
  
}
