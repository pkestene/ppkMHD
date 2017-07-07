#ifndef SDM_SHARED_H_
#define SDM_SHARED_H_

#include "shared/utils.h" // for UNUSED macro
#include "shared/enums.h"

namespace sdm {

//! Degree of freedom mapping inside a given cell for solution points.
//! the 3-uplet (i,j,k) identifies the location
//! index iv identifies the variable.
//! In 2d, of course, k is unused.
template<int dim, int N>
KOKKOS_INLINE_FUNCTION
constexpr int DofMap(int i, int j, int k, int iv) {

  return dim == 2 ? i + N*j + N*N*iv : i + N*j + N*N*k + N*N*N*iv;
  
} // DofMap

//! Degree of freedom mapping inside a given cell for flux points.
//! the 3-uplet (i,j,k) identifies the location
//! index iv identifies the variable.
//! In 2d, of course, k is unused.
//! @tparam dim is dimension (2 or 3)
//! @tparam N is the linear number of solution points per dim per cell
//! @tparam dir is the flux direction (IX, IY or IZ)
template<int dim, int N, int dir>
KOKKOS_INLINE_FUNCTION
int DofMapFlux(int i, int j, int k, int iv) {
  
  if (dir == IX) {
    
    return dim == 2 ?
      i + (N+1)*j + (N+1)*N*iv :
      i + (N+1)*j + (N+1)*N*k + (N+1)*N*N*iv;
    
  } else if (dir == IY) {
    
    return dim == 2 ?
      i +  N   *j + N*(N+1)*iv :
      i +  N   *j + N*(N+1)*k + N*(N+1)*N*iv;
    
  } else if (dir == IZ) {
    
    return dim == 2 ?
      0 :
      i + N    *j + N*N*k + N*N*(N+1)*iv;
    
  }

  // should never come here !
  return 0;
  
} // DofMapFlux

} // namespace sdm

#endif // SDM_SHARED_H_
