#ifndef SDM_SHARED_H_
#define SDM_SHARED_H_

#include "shared/utils.h" // for UNUSED macro

namespace sdm {

//! Degree of freedom mapping inside a given cell.
//! the 3-uplet (i,j,k) identifies the location
//! index iv identifies the variable.
//! In 2d, of course, k is unused.
template<int dim, int N>
KOKKOS_INLINE_FUNCTION
constexpr int DofMap(int i, int j, int k, int iv) {

  return dim == 2 ? i + N*j + N*N*iv : i + N*j + N*N*k + N*N*N*iv;
  
} // DofMap

} // namespace sdm

#endif // SDM_SHARED_H_
