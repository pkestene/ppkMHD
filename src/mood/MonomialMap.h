#ifndef MOOD_MONOMIAL_MAP_H_
#define MOOD_MONOMIAL_MAP_H_

#include <type_traits> // for std::integral_constant
#include "shared/real_type.h"
#include "shared/kokkos_shared.h"

#include "mood/monomials_ordering.h"
#include "mood/Binomial.h"

namespace mood {

/**
 * MonomialMap structure holds a 2D array:
 * - first entry is the monomial Id
 * - second entry multivariate index
 *
 */
struct MonomialMap {

  //! dimension : 2 or 3
  int dim;

  //! order (maximum degree of the multivariate polynomial)
  int order;
  
  //! total number of dim-variate monomials of order less than order.
  int Ncoefs;

  //! typedef for data map
  using MonomMap = Kokkos::View<int**, DEVICE>;
  using MonomMapHost = MonomMap::HostMirror; 
  
  //! store the exponent of each variable in a monomials, for all monomials.
  //int data[Ncoefs][dim];
  MonomMap data;
  MonomMapHost data_h;
  
  /**
   * Default constructor build monomials map entries.
   */
  MonomialMap(int dim, int order) :
    dim(dim),
    order(order),
    Ncoefs(binom(dim+order,dim)) {

    if (dim != 2 and dim != 3) {
      std::cerr << "[MonomialMap] error: invalid value for dim (should be 2 or 3)\n";
    }
    
    // memory allocation for map
    data   = MonomMap("data", Ncoefs, dim);
    data_h = Kokkos::create_mirror_view(data);

    if (dim == 2)
      init_map<2>();
    else
      init_map<3>();

    // upload data to device
    Kokkos::deep_copy(data,data_h);
    
  } // MonomialMap

  //! init map (on host and copy on device)
  template<int dim_>
  void init_map() {
      
    // exponent vector
    int e[dim_];
    for (int i=0; i<dim_; ++i) e[i] = 0;
    
    // d is the order, it will increase up to order
    int d = -1;
    
    int sum_e = 0;
    for (int i=0; i<dim_; ++i) sum_e += e[i];

    int index = 0;
    
    // span all possible monomials
    while ( sum_e <= order ) {

      if (dim_==2) {
	data_h(index,0) = e[0];
	data_h(index,1) = e[1];
      } else if (dim_ == 3) {
	data_h(index,0) = e[0];
	data_h(index,1) = e[1];
	data_h(index,2) = e[2];
      }
      
      // increment (in the sens of graded reverse lexicographic order)
      // the exponents vector representing a monomial x^e[0] * y^e[1] * z^[2]
      mono_next_grlex<dim_>(e);
      
      // update sum of exponents
      sum_e = 0;
      for (int i=0; i<dim_; ++i) sum_e += e[i];

      ++index;
      
    } // end while

  } // init_map
  
}; // class MonomialMap

} // namespace MonomialMap

#endif // MOOD_MONOMIAL_MAP_H_
