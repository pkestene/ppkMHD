#ifndef MONOMIALS_PRINT_UTILS_H_
#define MONOMIALS_PRINT_UTILS_H_

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <array>

#include "mood/monomials_ordering.h"
#include "mood/polynomials_utils.h"

namespace mood
{

/**
 * Print all monomials (ordered using Graded Reverse Lexicographic order)
 * up to a given order.
 *
 * \param order is the maximum order of a monomial.
 *
 * Let's remember that the monomial order is the sum of exponents.
 * If monomial is X^3 Y^2 Z^5, order is 3+2+5=10
 *
 * Exponent ordering
 * e = {e[2], e[1], e[0]}
 * e[0] is the least significant exponent (X_0 = X)
 * e[1] is                                (X_1 = Y)
 * e[2] is the most  significant exponent (X_2 = Z)
 */
template <unsigned int dim>
void
print_all_monomials(int order)
{

  std::cout << "#################################################\n";
  std::cout << "Multivariate monomials in dim = " << dim << " up to order " << order << "\n";
  std::cout << "#################################################\n";

  // exponent vector
  std::array<int, dim> e;
  for (int i = 0; i < dim; ++i)
    e[i] = 0;

  // d is the order, it will increase up to order
  int d = -1;

  int sum_e = 0;
  for (int i = 0; i < dim; ++i)
    sum_e += e[i];

  while (sum_e <= order)
  {

    // check if monomial order has been incremented
    if (sum_e > d)
    {
      d++;
      std::cout << "-- order " << d << "\n";
    }

    // print current monomial exponents, most significant exponent first
    if (dim == 2)
    {
      std::cout << " mono_next_grlex(dim=" << dim << ",e) = " << "(" << e[1] << "," << e[0]
                << ")\n";
    }
    else if (dim == 3)
    {
      std::cout << " mono_next_grlex(dim=" << dim << ",e) = " << "(" << e[2] << "," << e[1] << ","
                << e[0] << ")\n";
    }
    else if (dim == 4)
    {
      std::cout << " mono_next_grlex(dim=" << dim << ",e) = " << "(" << e[3] << "," << e[2] << ","
                << e[1] << "," << e[0] << ")\n";
    }

    // increment (in the sens of graded reverse lexicographic order)
    // the exponents vector representing a monomial x^e[0] * y^e[1] * z^[2]
    mono_next_grlex<dim>(e);

    // update sum of exponents
    sum_e = 0;
    for (int i = 0; i < dim; ++i)
      sum_e += e[i];
  }

} // print_all_monomials


template <unsigned int dim>
void
print_all_monomials_map(int order)
{

  std::cout << "#################################################\n";
  std::cout << "Multivariate monomials in dim = " << dim << " up to order " << order << "\n";
  std::cout << "#################################################\n";

  // what is the total number of monomials ?
  // (dim+order)! / dim! / order!
  if (dim == 2)
    std::cout << "int monomials_map[" << get_number_of_coefficients<2>(order) << "] = {\n";
  else if (dim == 3)
    std::cout << "int monomials_map[" << get_number_of_coefficients<3>(order) << "] = {\n";

  // exponent vector
  std::array<int, dim> e;
  for (int i = 0; i < dim; ++i)
    e[i] = 0;

  int sum_e = 0;
  for (int i = 0; i < dim; ++i)
    sum_e += e[i];

  // span all possible monomials
  while (sum_e <= order)
  {

    std::cout << "    {";

    if (dim == 2)
    {

      std::cout << e[0] << "," << e[1] << "},";
      std::cout << "   // " << "X^" << e[0] << " * " << "Y^" << e[1] << "\n";
    }
    else if (dim == 3)
    {

      std::cout << e[0] << "," << e[1] << "," << e[2] << "},";
      std::cout << "   // " << "X^" << e[0] << " * " << "Y^" << e[1] << " * " << "Z^" << e[2]
                << "\n";
    }

    // increment (in the sens of graded reverse lexicographic order)
    // the exponents vector representing a monomial x^e[0] * y^e[1] * z^[2]
    mono_next_grlex<dim>(e);

    // update sum of exponents
    sum_e = 0;
    for (int i = 0; i < dim; ++i)
      sum_e += e[i];
  }

  // close the array
  std::cout << "};\n";

} // print_all_monomials_map

} // namespace mood

#endif // MONOMIALS_PRINT_UTILS_H_
