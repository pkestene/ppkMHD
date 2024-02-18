#ifndef MOOD_MONOMIAL_MAP_H_
#define MOOD_MONOMIAL_MAP_H_

#include <type_traits> // for std::integral_constant
#include "shared/real_type.h"
#include "shared/kokkos_shared.h"

#include "mood/monomials_ordering.h"
#include "mood/Binomial.h"

namespace mood
{

/**
 * MonomialMap structure holds a 2D array:
 * - first entry is the monomial Id
 * - second entry multivariate index
 *
 * template parameters are
 * dim : dimension (either 2 or 3)
 * degree : degree of the polynomial (maximum degree of the multivariate polynomial)
 */
template <int dim, int degree>
class MonomialMap
{

public:
  //! total number of dim-variate monomials of order less than order.
  static constexpr int ncoefs = mood::binomial<degree + dim, degree>();

  //! typedef for data map
  using MonomMap = Kokkos::View<int[ncoefs][dim], Device>;
  using MonomMapHost = typename MonomMap::HostMirror;

  //! store the exponent of each variable in a monomials, for all monomials.
  // int data[ncoefs][dim];
  MonomMap     data;
  MonomMapHost data_h;

  /**
   * Default constructor build monomials map entries.
   */
  MonomialMap()
  {

    if (dim != 2 and dim != 3)
    {
      std::cerr << "[MonomialMap] error: invalid value for dim (should be 2 or 3)\n";
    }

    // memory allocation for map
    data = MonomMap("data");
    data_h = Kokkos::create_mirror_view(data);

    init_map();

    // upload data to device
    Kokkos::deep_copy(data, data_h);

  } // MonomialMap

  virtual ~MonomialMap(){};

  //! init map (on host and copy on device)
  void
  init_map()
  {

    // exponent vector
    std::array<int, dim> e;
    for (int i = 0; i < e.size(); ++i)
      e[i] = 0;

    int sum_e = 0;
    for (int i = 0; i < e.size(); ++i)
      sum_e += e[i];

    int index = 0;

    // span all possible monomials
    while (sum_e <= degree)
    {

      if (dim == 2)
      {
        data_h(index, 0) = e[0];
        data_h(index, 1) = e[1];
      }
      else if (dim == 3)
      {
        data_h(index, 0) = e[0];
        data_h(index, 1) = e[1];
        data_h(index, 2) = e[2];
      }

      // increment (in the sens of graded reverse lexicographic order)
      // the exponents vector representing a monomial x^e[0] * y^e[1] * z^[2]
      mono_next_grlex<dim>(e);

      // update sum of exponents
      sum_e = 0;
      for (int i = 0; i < dim; ++i)
        sum_e += e[i];

      ++index;

    } // end while

  } // init_map

}; // class MonomialMap

} // namespace mood

#endif // MOOD_MONOMIAL_MAP_H_
