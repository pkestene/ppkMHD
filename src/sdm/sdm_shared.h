#ifndef SDM_SHARED_H_
#define SDM_SHARED_H_

#include "shared/utils.h" // for UNUSED macro
#include "shared/enums.h"

namespace sdm {

/**
 * from global index to local cell Id and Dof id
 *
 * \param[in] ii global index
 * \param[out] i cell coordinate index
 * \param[out] idx Dof index (solution)
 */
KOKKOS_INLINE_FUNCTION
void global2local(int ii,   int jj,
                  int &i,   int &j,
                  int &idx, int &idy,
                  int N)
{

    i = ii / N;
    j = jj / N;
    
    idx = ii-i*N;
    idy = jj-j*N;

} // global2local - 2d

/**
 * from global index to local cell Id and Dof id
 *
 * \param[in] ii global index
 * \param[out] i cell coordinate index
 * \param[out] idx Dof index (solution)
 */
KOKKOS_INLINE_FUNCTION
void global2local(int ii,   int jj,   int kk,
                  int &i,   int &j,   int &k,
                  int &idx, int &idy, int &idz,
                  int N)
{
  
  i = ii / N;
  j = jj / N;
  k = kk / N;
  
  idx = ii - i * N;
  idy = jj - j * N;
  idz = kk - k * N;
  
} // global2local - 3d

/**
 * from global index to local cell Id and Dof id
 *
 * \param[in] ii global index
 * \param[out] i cell coordinate index
 * \param[out] idx Dof index (flux)
 */
template<int dir>
KOKKOS_INLINE_FUNCTION
void global2local_flux(int ii,   int jj,
                       int &i,   int &j,
                       int &idx, int &idy,
                       int N)
{
  
  if (dir == IX) {
    
    i = ii / (N + 1);
    j = jj / N;
    
    idx = ii-i*(N+1);
    idy = jj-j*N;
    
  } else {
    
    i = ii / N;
    j = jj / (N + 1);
    
    idx = ii-i*N;
    idy = jj-j*(N+1);
    
  }
  
} // global2local_flux

/**
 * from global index to local cell Id and Dof id
 *
 * \param[in] ii global index
 * \param[out] i cell coordinate index
 * \param[out] idx Dof index (flux)
 */
template<int dir>
KOKKOS_INLINE_FUNCTION
void global2local_flux(int ii,   int jj,   int kk,
                       int &i,   int &j,   int &k,
                       int &idx, int &idy, int &idz,
                       int N)
{
  
  if (dir == IX) {
    
    i = ii / (N + 1);
    j = jj / N;
    k = kk / N;
    
    idx = ii-i*(N+1);
    idy = jj-j*N;
    idz = kk-k*N;
    
  } else if (dir == IY) {
    
    i = ii / N;
    j = jj / (N + 1);
    k = kk / N;
    
    idx = ii-i*N;
    idy = jj-j*(N+1);
    idz = kk-k*N;
    
  } else {
    
    i = ii / N;
    j = jj / N;
    k = kk / (N + 1);
    
    idx = ii-i*N;
    idy = jj-j*N;
    idz = kk-k*(N+1);
    
  }
  
} // global2local_flux

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
    
  } else {

    // should never come here !
    return 0;

  }
  
} // DofMapFlux

} // namespace sdm

#endif // SDM_SHARED_H_
