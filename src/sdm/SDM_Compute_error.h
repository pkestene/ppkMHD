#ifndef SDM_COMPUTE_ERROR_FUNCTOR_H_
#define SDM_COMPUTE_ERROR_FUNCTOR_H_

#include <limits> // for std::numeric_limits
#ifdef __CUDA_ARCH__
#  include <math_constants.h> // for cuda math constants, e.g. CUDART_INF
#endif                        // __CUDA_ARCH__

#include "shared/kokkos_shared.h"
#include "sdm/SDMBaseFunctor.h"

#include "sdm/SDM_Geometry.h"
#include "sdm/sdm_shared.h" // for DofMap

#include "shared/EulerEquations.h"

namespace ppkMHD
{
namespace sdm
{

enum norm_type
{
  NORM_L1,
  NORM_L2
};

/*************************************************/
/*************************************************/
/*************************************************/
/**
 * compute L1 / L2 error between two data array (solution data).
 *
 * \tparam N is the SDM scheme order (nb of point per direction per cell)
 * \tparam norm should take value in NORM_L1 / NORM_L2
 *
 */
template <int N, int norm>
class Compute_Error_Functor_2d : public SDMBaseFunctor<2, N>
{

public:
  using typename SDMBaseFunctor<2, N>::DataArray;
  using typename SDMBaseFunctor<2, N>::HydroState;

  //! intra-cell degrees of freedom mapping at solution points
  static constexpr auto dofMap = DofMap<2, N>;

  /**
   * \param[in] varId identify which variable to reduce (ID, IE, IU, ...)
   */
  Compute_Error_Functor_2d(HydroParams        params,
                           SDM_Geometry<2, N> sdm_geom,
                           DataArray          Udata1,
                           DataArray          Udata2,
                           int                varId)
    : SDMBaseFunctor<2, N>(params, sdm_geom)
    , Udata1(Udata1)
    , Udata2(Udata2)
    , varId(varId){};

  // static method which does it all: create and execute functor
  static double
  apply(HydroParams        params,
        SDM_Geometry<2, N> sdm_geom,
        DataArray          Udata1,
        DataArray          Udata2,
        int                varId)
  {
    int64_t nbCells = params.isize * params.jsize;

    real_t                            error = 0;
    Compute_Error_Functor_2d<N, norm> functor(params, sdm_geom, Udata1, Udata2, varId);
    Kokkos::parallel_reduce(nbCells, functor, error);
    return error;
  }

  //! dummy trick
  static double
  apply(HydroParams        params,
        SDM_Geometry<3, N> sdm_geom,
        DataArray3d        Udata1,
        DataArray3d        Udata2,
        int                varId)
  {
    return -1.0;
  }

  // Tell each thread how to initialize its reduction result.
  KOKKOS_INLINE_FUNCTION
  void
  init(real_t & dst) const
  {
    // The identity under '+' is zero.
    // Kokkos does not come with a portable way to access
    // floating-point Inf and NaN.
    dst = 0.0;
  } // init

  // ================================================
  //
  // 2D version.
  //
  // ================================================
  //! functor for 2d
  KOKKOS_INLINE_FUNCTION
  void
  operator()(const int & index, real_t & sum) const
  {
    const int isize = this->params.isize;
    const int jsize = this->params.jsize;
    const int ghostWidth = this->params.ghostWidth;

    // local cell index
    int i, j;
    index2coord(index, i, j, isize, jsize);

    if (j >= ghostWidth && j < jsize - ghostWidth && i >= ghostWidth && i < isize - ghostWidth)
    {

      // loop over current cell DoF solution points
      for (int idy = 0; idy < N; ++idy)
      {
        for (int idx = 0; idx < N; ++idx)
        {

          // get local conservative variable
          real_t tmp1 = Udata1(i, j, dofMap(idx, idy, 0, varId));
          real_t tmp2 = Udata2(i, j, dofMap(idx, idy, 0, varId));

          if (norm == NORM_L1)
          {
            sum += fabs(tmp1 - tmp2);
          }
          else
          {
            sum += (tmp1 - tmp2) * (tmp1 - tmp2);
          }

        } // end for idx
      } // end for idy

    } // end guard - ghostcells

  } // end operator () - 2d

  // "Join" intermediate results from different threads.
  // This should normally implement the same reduction
  // operation as operator() above. Note that both input
  // arguments MUST be declared volatile.
  KOKKOS_INLINE_FUNCTION
#if KOKKOS_VERSION_MAJOR > 3
  void
  join(real_t & dst, const real_t & src) const
#else
  void
  join(volatile real_t & dst, const volatile real_t & src) const
#endif
  {
    // + reduce
    dst += src;
  } // join

  DataArray Udata1;
  DataArray Udata2;
  int       varId;

}; // class Compute_Error_Functor_2d

/*************************************************/
/*************************************************/
/*************************************************/
/**
 * compute L1 / L2 error between two data array (solution data).
 *
 * \tparam N is the SDM scheme order (nb of point per direction per cell)
 * \tparam norm should take value in NORM_L1 / NORM_L2
 *
 * 3d version.
 */
template <int N, int norm>
class Compute_Error_Functor_3d : public SDMBaseFunctor<3, N>
{

public:
  using typename SDMBaseFunctor<3, N>::DataArray;
  using typename SDMBaseFunctor<3, N>::HydroState;

  //! intra-cell degrees of freedom mapping at solution points
  static constexpr auto dofMap = DofMap<3, N>;

  /**
   * \param[in] varId identify which variable to reduce (ID, IE, IU, ...)
   */
  Compute_Error_Functor_3d(HydroParams        params,
                           SDM_Geometry<3, N> sdm_geom,
                           DataArray          Udata1,
                           DataArray          Udata2,
                           int                varId)
    : SDMBaseFunctor<3, N>(params, sdm_geom)
    , Udata1(Udata1)
    , Udata2(Udata2)
    , varId(varId){};

  // static method which does it all: create and execute functor
  static double
  apply(HydroParams        params,
        SDM_Geometry<3, N> sdm_geom,
        DataArray          Udata1,
        DataArray          Udata2,
        int                varId)
  {
    int64_t nbCells = params.isize * params.jsize * params.ksize;

    real_t                            error = 0;
    Compute_Error_Functor_3d<N, norm> functor(params, sdm_geom, Udata1, Udata2, varId);
    Kokkos::parallel_reduce(nbCells, functor, error);
    return error;
  }

  //! dummy trick
  static double
  apply(HydroParams        params,
        SDM_Geometry<2, N> sdm_geom,
        DataArray2d        Udata1,
        DataArray2d        Udata2,
        int                varId)
  {
    return -1.0;
  }

  // Tell each thread how to initialize its reduction result.
  KOKKOS_INLINE_FUNCTION
  void
  init(real_t & dst) const
  {
    // The identity under '+' is zero.
    // Kokkos does not come with a portable way to access
    // floating-point Inf and NaN.
    dst = 0.0;
  } // init

  // ================================================
  //
  // 3D version.
  //
  // ================================================
  //! functor for 3d
  KOKKOS_INLINE_FUNCTION
  void
  operator()(const int & index, real_t & sum) const
  {
    const int isize = this->params.isize;
    const int jsize = this->params.jsize;
    const int ksize = this->params.ksize;
    const int ghostWidth = this->params.ghostWidth;

    // local cell index
    int i, j, k;
    index2coord(index, i, j, k, isize, jsize, ksize);

    if (k >= ghostWidth and k < ksize - ghostWidth and j >= ghostWidth and
        j < jsize - ghostWidth and i >= ghostWidth and i < isize - ghostWidth)
    {

      // loop over current cell DoF solution points
      for (int idz = 0; idz < N; ++idz)
      {
        for (int idy = 0; idy < N; ++idy)
        {
          for (int idx = 0; idx < N; ++idx)
          {

            // get local conservative variable
            real_t tmp1 = Udata1(i, j, k, dofMap(idx, idy, idz, varId));
            real_t tmp2 = Udata2(i, j, k, dofMap(idx, idy, idz, varId));

            if (norm == NORM_L1)
            {
              sum += fabs(tmp1 - tmp2);
            }
            else
            {
              sum += (tmp1 - tmp2) * (tmp1 - tmp2);
            }

          } // end for idx
        } // end for idy
      } // end for idz

    } // end guard - ghostcells

  } // end operator () - 3d

  // "Join" intermediate results from different threads.
  // This should normally implement the same reduction
  // operation as operator() above. Note that both input
  // arguments MUST be declared volatile.
  KOKKOS_INLINE_FUNCTION
#if KOKKOS_VERSION_MAJOR > 3
  void
  join(real_t & dst, const real_t & src) const
#else
  void
  join(volatile real_t & dst, const volatile real_t & src) const
#endif
  {
    // + reduce
    dst += src;
  } // join

  DataArray Udata1;
  DataArray Udata2;
  int       varId;

}; // class Compute_Error_Functor_3d

} // namespace sdm
} // namespace ppkMHD

#endif // SDM_COMPUTE_ERROR_FUNCTOR_H_
