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


  /*
   * test class Polynomial.
   */
  std::cout << "############################\n";
  std::cout << "Testing class Polynomial    \n";
  std::cout << "############################\n";
  constexpr int ncoefs = mood::binomial<3+2,3>();
  std::array<real_t,ncoefs> data;
  for (int i=0; i<ncoefs; ++i)
    data[i] = 1.0*i;
  
  using Polynomial_t = mood::Polynomial<3,2>;
  Polynomial_t polynomial(data);

  std::cout << " eval at (1.0, 1.0, -1.0) " << polynomial.eval(1.0, 1.0, -1.0) << "\n";
  
  for (int i = 0; i<Polynomial_t::Ncoefs; ++i) {
    int e[3] = {Polynomial_t::monomialMap.data[i][0],
		Polynomial_t::monomialMap.data[i][1],
		Polynomial_t::monomialMap.data[i][2]};

    std::cout << polynomial.getCoefs(i);
    std::cout << "    {";
    std::cout << e[0] << "," << e[1] << "," << e[2] << "},";
    std::cout << "   // " << "X^" << e[0] << " * " << "Y^" << e[1] << " * " << "Z^" << e[2] << "\n";

  }
  
  return 0;
  
}
