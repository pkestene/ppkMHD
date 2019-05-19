#ifndef SDM_SHARED_H_
#define SDM_SHARED_H_

#include "shared/utils.h" // for UNUSED macro
#include "shared/enums.h"

namespace sdm {

//! Main data structure
using DataArray = Kokkos::View<real_t***, Kokkos::LayoutLeft, Device>;

//! Data array typedef for host memory space
using DataArrayHost = DataArray::HostMirror;

//! unmanaged view
using DataArrayUnmanaged = Kokkos::View<real_t***, Kokkos::LayoutLeft, Device, Kokkos::MemoryTraits<Kokkos::Unmanaged> >;

//! data structure for cell average data
using DataArrayAv2d = Kokkos::View<real_t***, Kokkos::LayoutLeft, Device>;

//! data structure for cell average data
using DataArrayAv3d = Kokkos::View<real_t****, Kokkos::LayoutLeft, Device>;

/**
 * Utility to convert a global index to iDof,iCell according to
 * index = iDof + N**d * iCell
 *
 * \param[in] index
 * \param[in] SDM order
 * \param[out] iDof Degree of freedom index
 * \param[out] iCell cell index
 */
KOKKOS_INLINE_FUNCTION
void index_to_iDof_iCell(int64_t index, int Nd, int &iDof, int &iCell)
{

  iCell = index/Nd;
  iDof = index - Nd*iCell;

}; // index_to_iDof_iCell


/**
 * Convert iDof to dof coordinates according to
 * iDof = idx + N * idy
 */
KOKKOS_INLINE_FUNCTION
void iDof_to_coord(int iDof, int N, int& idx, int& idy)
{
  idy = iDof/N;
  idx = iDof-N*idy;
}; // iDof_to_coord

/**
 * Convert iDof to dof coordinates according to
 * iDof = idx + N * (idy + N*idz)
 */
KOKKOS_INLINE_FUNCTION
void iDof_to_coord(int iDof, int N, int& idx, int& idy, int& idz)
{
  idz = iDof/(N*N);
  const int tmp = iDof - N*N*idz;
  idy = tmp/N;
  idx = tmp-N*idy;
}; // iDof_to_coord

/**
 * Convert iDof to dof coordinates according to
 * iDof = idx + (N+1) * idy if dir == IX
 * iDof = idx + N * idy if dir == IY
 */
template<int dir>
KOKKOS_INLINE_FUNCTION
void iDof_to_coord_flux(int iDof, int N, int& idx, int& idy)
{
  if (dir==IX) {
    idy = iDof / (N+1);
    idx = iDof - (N+1) * idy;
  } else {
    idy = iDof / N;
    idx = iDof - N * idy;
  }
}; // iDof_to_coord_flux

/**
 * Convert iDof to dof coordinates according to
 * iDof = idx + (N+1) * (idy + N*idz)     if dir==IX
 * iDof = idx +  N    * (idy + (N+1)*idz) if dir==IY
 * iDof = idx +  N    * (idy +  N   *idz) if dir==IZ
 */
 template<int dir>
KOKKOS_INLINE_FUNCTION
void iDof_to_coord_flux(int iDof, int N, int& idx, int& idy, int& idz)
{
  if (dir==IX) {
    idz = iDof / ((N+1) * N);
    const int tmp = iDof - (N+1) * N * idz;
    idy = tmp / (N+1);
    idx = tmp - (N+1) * idy;
  } else if (dir==IY) {
    idz = iDof / (N * (N+1));
    const int tmp = iDof - N * (N+1) * idz;
    idy = tmp / N;
    idx = tmp - N * idy;
  } else {
    idz = iDof / (N * N);
    const int tmp = iDof - N * N * idz;
    idy = tmp / N;
    idx = tmp - N * idy;
  }
}; // iDof_to_coord


/**
 * Convert iCell to cell coordinates according to
 * iCell = i + isize * j
 */
KOKKOS_INLINE_FUNCTION
void iCell_to_coord(int iCell, int isize, int& i, int& j)
{
  j = iCell/isize;
  i = iCell-isize*j;
}; // iDof_to_coord

/**
 * Convert iDof to dof coordinates according to
 * iCell = i + isize * (j + jsize * k)
 */
KOKKOS_INLINE_FUNCTION
void iCell_to_coord(int iCell, int isize, int jsize,
                    int& i, int& j, int& k)
{
  k = iCell/(isize*jsize);
  const int tmp = iCell - isize*jsize*k;
  j = tmp/isize;
  i = tmp-isize*j;
}; // iDof_to_coord

/**
 * Convert coordinates to iDof according to
 * iDof = idx + N * idy
 */
KOKKOS_INLINE_FUNCTION
int coord_to_iDof(int idx, int idy, int N)
{
  return idx+N*idy;
}; // coord_to_iDof

/**
 * Convert coordinates to iDof according to
 * iDof = idx + N * idy + N*N*idz
 */
KOKKOS_INLINE_FUNCTION
int coord_to_iDof(int idx, int idy, int idz, int N)
{
  return idx+N*idy+N*N*idz;
}; // coord_to_iDof

/**
 * Convert coordinates to iCell according to
 * iCell = i + isize * j
 */
KOKKOS_INLINE_FUNCTION
int coord_to_iCell(int i, int j, int isize)
{
  return i + isize*j;
}; // coord_to_iCell

/**
 * Convert coordinates to iCell according to
 * iCell = i + isize * j
 */
KOKKOS_INLINE_FUNCTION
int coord_to_iCell(int i, int j, int k, int isize, int jsize)
{
  return i + isize*j + isize*jsize*k;
}; // coord_to_iCell

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
